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


def remap_checkpoint_dataset(
    state_dict: dict,
    dataset_remapping: dict[str, str],
) -> dict:
    """Remap dataset names in a checkpoint state_dict.

    Handles renaming dataset-specific layers (encoders, decoders,
    pre/post processors, node attributes) from old to new dataset names.

    Parameters
    ----------
    state_dict : dict
        The checkpoint state dict to remap.
    dataset_remapping : dict[str, str]
        Mapping from old dataset name to new dataset name.
        e.g. {'data': 'era5'} renames all 'data' layers to 'era5'.

    Returns
    -------
    dict
        Remapped state dict.
    """
    # Dataset-specific layer prefixes to remap
    dataset_prefixes = [
        "model.pre_processors.",
        "model.post_processors.",
        "model.pre_processors_tendencies.",
        "model.post_processors_tendencies.",
        "model.model.encoder.",
        "model.model.encoder_graph_provider",
        "model.model.decoder.",
        "model.model.decoder_graph_provider",
        "model.model.node_attributes.",
    ]

    remapped, unchanged = {}, []
    rule_hits: dict[str, int] = {}
    new_state_dict = {}

    for old_key, value in state_dict.items():
        new_key = old_key

        for prefix in dataset_prefixes:
            for old_name, new_name in dataset_remapping.items():
                pattern = f"{prefix}{old_name}."
                replacement = f"{prefix}{new_name}."
                if pattern in new_key:
                    new_key = new_key.replace(pattern, replacement)
                    rule_hits[f"{pattern} -> {replacement}"] = rule_hits.get(f"{pattern} -> {replacement}", 0) + 1
                    break  # only one dataset name can match per prefix

        if new_key != old_key:
            remapped[old_key] = new_key
        else:
            unchanged.append(old_key)

        new_state_dict[new_key] = value

    # Summary logging
    LOGGER.info(
        "Checkpoint dataset remapping: %d keys remapped, %d unchanged (of %d total).",
        len(remapped),
        len(unchanged),
        len(state_dict),
    )
    for rule, count in rule_hits.items():
        LOGGER.info("  %s: %d keys", rule, count)

    LOGGER.debug("Example remapped keys (up to 10):")
    for old_k, new_k in list(remapped.items())[:10]:
        LOGGER.debug("  %s -> %s", old_k, new_k)

    return new_state_dict


def transfer_learning_loading(
    model: torch.nn.Module, 
    ckpt_path: Path | str,
    dataset_remapping: dict[str, str] | None = None,
) -> nn.Module:
    # Load the checkpoint
    LOGGER.debug("Loading checkpoint to device: %s", model.device)
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=model.device)

    # apply chunking migration (fails silently otherwise leading to hard to debug issues)
    # this is due to loading with strict=False, planning to make this more robust in the future
    checkpoint = chunking_fix_migration(checkpoint)

    # Refresh processor stats from the current dataset if configured.
    model._update_checkpoint_state_dict_for_load(checkpoint)

    # Filter out layers with size mismatch
    state_dict = checkpoint["state_dict"]
    
    # Remap dataset names in state_dict before loading
    if dataset_remapping:
        LOGGER.info("Applying dataset remapping: %s", dataset_remapping)
        state_dict = remap_checkpoint_dataset(state_dict, dataset_remapping)

    model_state_dict = model.state_dict()

    for key in state_dict.copy():
        if key in model_state_dict and state_dict[key].shape != model_state_dict[key].shape:
            LOGGER.info("Skipping loading parameter: %s", key)
            LOGGER.info("Checkpoint shape: %s", str(state_dict[key].shape))
            LOGGER.info("Model shape: %s", str(model_state_dict[key].shape))

            del state_dict[key]  # Remove the mismatched key

    # Load the filtered st-ate_dict into the model
    model.load_state_dict(state_dict, strict=False)

    ## Needed for data indices check
    data_indices = checkpoint["hyper_parameters"]["data_indices"]

    if isinstance(data_indices, dict):
        # New format: data_indices is always a dict in new code (even for single-dataset)
        LOGGER.info("Loading checkpoint with datasets: %s", list(data_indices.keys()))
        model._ckpt_model_name_to_index = {
            dataset_name: indices.name_to_index for dataset_name, indices in data_indices.items()
        }
    else:
        # Old format: data_indices is a single IndexCollection object (not dict)
        msg = (
            f"Checkpoint at '{ckpt_path}' was created with an older version of anemoi-core "
            "that does not support multi-dataset training. This checkpoint is incompatible "
            "with transfer learning in the current version."
        )
        raise TypeError(msg)

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
