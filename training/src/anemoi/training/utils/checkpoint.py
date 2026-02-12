# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import importlib
import io
import logging
import pickle
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from pytorch_lightning import Callback
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer

from anemoi.models.migrations import Migrator
from anemoi.training.train.tasks.base import BaseGraphModule
from anemoi.utils.checkpoints import save_metadata

chunking_fix_migration = importlib.import_module("anemoi.models.migrations.scripts.1762857428_chunking_fix").migrate

LOGGER = logging.getLogger(__name__)


def load_and_prepare_model(lightning_checkpoint_path: str) -> tuple[torch.nn.Module, dict]:
    """Load the lightning checkpoint and extract the pytorch model and its metadata.

    Parameters
    ----------
    lightning_checkpoint_path : str
        path to lightning checkpoint

    Returns
    -------
    tuple[torch.nn.Module, dict]
        pytorch model, metadata

    """
    module = BaseGraphModule.load_from_checkpoint(lightning_checkpoint_path, weights_only=False)
    model = module.model

    metadata = dict(**model.metadata)
    model.metadata = None
    model.config = None

    return model, metadata


def save_inference_checkpoint(model: torch.nn.Module, metadata: dict, save_path: Path | str) -> Path:
    """Save a pytorch checkpoint for inference with the model metadata.

    Parameters
    ----------
    model : torch.nn.Module
        Pytorch model
    metadata : dict
        Anemoi Metadata to inject into checkpoint
    save_path : Path | str
        Directory to save anemoi checkpoint

    Returns
    -------
    Path
        Path to saved checkpoint
    """
    save_path = Path(save_path)
    inference_filepath = save_path.parent / f"inference-{save_path.name}"

    torch.save(model, inference_filepath)
    save_metadata(inference_filepath, metadata)
    return inference_filepath

def get_trainable_key(param_name: str) -> str | None:
    """
    Helper function used when transfer learning to identify changes in trainable_parameters numbers.
    """
    if ".encoder." in param_name:
        return "data2hidden"
    if ".decoder." in param_name:
        return "hidden2data"
    if ".processor." in param_name:
        return "hidden2hidden"
    if ".data." in param_name:
        return "data"
    if ".hidden." in param_name:
        return "hidden"
    return None

def transfer_learning_loading(model: torch.nn.Module, ckpt_path: Path | str, model_config) -> nn.Module:
    # Load the checkpoint
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=model.device)

    # apply chunking migration (fails silently otherwise leading to hard to debug issues)
    # this is due to loading with strict=False, planning to make this more robust in the future
    checkpoint = chunking_fix_migration(checkpoint)

    # extract trainable_parameters dictionary from the model config
    trainable_parameters = model_config.trainable_parameters

    # check whether sizes of components are compatible, either matching or differing by trainable_parameters
    state_dict = checkpoint["state_dict"]
    model_state_dict = model.state_dict()

    for key in list(state_dict.keys()):
        if key not in model_state_dict:
            continue

        ckpt_tensor = state_dict[key]
        model_tensor = model_state_dict[key]

        if ckpt_tensor.shape == model_tensor.shape:
            continue  # perfect match

        if ckpt_tensor.ndim != model_tensor.ndim:
            LOGGER.info("Skipping %s (different ndim)", key)
            del state_dict[key]
            continue

        # check whether the size of the parameter grows by the number of trainable parameters
        # if so, load it into the matching slice of the tensor
        growth_key = get_trainable_key(key)
        allowed_growth = trainable_parameters.get(growth_key, None)

        if allowed_growth is None:
            LOGGER.info("Skipping %s (no growth rule)", key)
            del state_dict[key]
            continue

        # compute per-dimension differences
        diffs = [m - c for c, m in zip(ckpt_tensor.shape, model_tensor.shape)]

        # only allow change in parameter size in ONE dimension equal to allowed_growth
        # if checkpoint parameter has shape [num_channels, size], model has [num_channels, size + allowed_growth]
        # then can load weights into first [num_channels, size] of the model weights 
        # i.e. only the trainable_parameters are initialised from scratch
        positive_diffs = [d for d in diffs if d > 0]

        if positive_diffs == [allowed_growth] and all(d >= 0 for d in diffs):
            LOGGER.info("Partially loading %s with allowed growth %d from key %s", key, allowed_growth, growth_key)
            LOGGER.info("Checkpoint shape: %s", tuple(ckpt_tensor.shape))
            LOGGER.info("Model shape: %s", tuple(model_tensor.shape))

            new_tensor = model_tensor.clone()
            slices = tuple(slice(0, min(c, m)) for c, m in zip(ckpt_tensor.shape, model_tensor.shape))
            new_tensor[slices] = ckpt_tensor[slices]
            state_dict[key] = new_tensor
        else:
            LOGGER.info("Skipping %s (shape change not matching config)", key)
            LOGGER.info("Checkpoint shape: %s", tuple(ckpt_tensor.shape))
            LOGGER.info("Model shape: %s", tuple(model_tensor.shape))
            del state_dict[key]

    # Load the filtered st-ate_dict into the model
    model.load_state_dict(state_dict, strict=False)
    # Needed for data indices check
    # Handle both single-dataset and multi-dataset checkpoints
    try:
        # Try multi-dataset format first
        data_indices = checkpoint["hyper_parameters"]["data_indices"]
        if isinstance(data_indices, dict):
            model._ckpt_model_name_to_index = {
                dataset_name: indices.name_to_index
                for dataset_name, indices in data_indices.items()
            }
        else:
            model._ckpt_model_name_to_index = data_indices.name_to_index
    except (KeyError, AttributeError):
        # Fall back to single-dataset format for older checkpoints
        model._ckpt_model_name_to_index = checkpoint["hyper_parameters"]["data_indices"].name_to_index
    return model


def freeze_submodule_by_name(module: nn.Module, target_name: str) -> None:
    """Recursively freezes the parameters of a submodule with the specified name.

    Parameters
    ----------
    module : torch.nn.Module
        Pytorch model
    target_name : str
        The name of the submodule to freeze.
    """
    for name, child in module.named_children():
        # If this is the target submodule, freeze its parameters
        if name == target_name:
            for param in child.parameters():
                param.requires_grad = False
        else:
            # Recursively search within children
            freeze_submodule_by_name(child, target_name)


class LoggingUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> str:
        if "anemoi.training" in module:
            msg = (
                f"anemoi-training Pydantic schemas found in model's metadata: "
                f"({module}, {name}) Please review Pydantic schemas to avoid this."
            )
            raise ValueError(msg)
        return super().find_class(module, name)


def check_classes(model: torch.nn.Module) -> None:
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)
    _ = LoggingUnpickler(buffer).load()


class RegisterMigrations(Callback):
    """Callback that register all existing migrations to a checkpoint before storing it."""

    def __init__(self):
        self.migrator = Migrator()

    def on_save_checkpoint(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        checkpoint: dict[str, Any],
    ) -> None:
        self.migrator.register_migrations(checkpoint)
