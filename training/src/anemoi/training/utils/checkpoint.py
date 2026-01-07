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
from collections import defaultdict

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

def remap_single_to_multidataset(checkpoint_state_dict: dict, dataset_name: str = 'era5', data_node_name: str = 'data') -> dict:
    LOGGER.info("Starting key remapping from single-dataset to multi-dataset format...")

    updated_state_dict = {}

    # Counters for logging
    rule_hits = defaultdict(int)
    unchanged_keys = []
    remapped_keys = []

    for old_key, value in checkpoint_state_dict.items():
        new_key = old_key

        # --- pre/post processors (data) ---
        if "model.pre_processors.data." in new_key:
            new_key = new_key.replace(
                "model.pre_processors.data.",
                f"model.pre_processors.{dataset_name}."
            )
            rule_hits[f"pre_processors.data -> pre_processors.{dataset_name}"] += 1

        if "model.post_processors.data." in new_key:
            new_key = new_key.replace(
                "model.post_processors.data.",
                f"model.post_processors.{dataset_name}."
            )
            rule_hits[f"post_processors.data -> post_processors.{dataset_name}"] += 1

        # --- pre/post processors (tendencies) ---
        if "model.pre_processors_tendencies." in new_key:
            new_key = new_key.replace(
                "model.pre_processors_tendencies.",
                f"model.pre_processors_tendencies.{dataset_name}."
            )
            rule_hits[f"pre_processors_tendencies -> *.{dataset_name}"] += 1

        if "model.post_processors_tendencies." in new_key:
            new_key = new_key.replace(
                "model.post_processors_tendencies.",
                f"model.post_processors_tendencies.{dataset_name}."
            )
            rule_hits[f"post_processors_tendencies -> *.{dataset_name}"] += 1

        # --- encoder / decoder ---
        if "model.model.encoder.data." in new_key:
            new_key = new_key.replace(
                "model.model.encoder.data.",
                f"model.model.encoder.{dataset_name}."
            )
            rule_hits[f"encoder.data -> encoder.{dataset_name}"] += 1

        if "model.model.decoder.data." in new_key:
            new_key = new_key.replace(
                "model.model.decoder.data.",
                f"model.model.decoder.{dataset_name}."
            )
            rule_hits[f"decoder.data -> decoder.{dataset_name}"] += 1

        # --- node attributes ---
        # Handle data node attributes with specific name change
        if "model.model.node_attributes.data.latlons_data" in new_key:
            new_key = new_key.replace(
                "model.model.node_attributes.data.latlons_data",
                f"model.model.node_attributes.{dataset_name}.latlons_{data_node_name}"
            )
            rule_hits[f"node_attributes latlons_data -> latlons_{data_node_name}"] += 1
        
        # Handle hidden node attributes (name stays the same)
        elif "model.model.node_attributes.data.latlons_hidden" in new_key:
            new_key = new_key.replace(
                "model.model.node_attributes.data.latlons_hidden",
                f"model.model.node_attributes.{dataset_name}.latlons_hidden"
            )
            rule_hits[f"node_attributes latlons_hidden -> latlons_hidden"] += 1
        
        # Catch-all for other node attributes
        elif "model.model.node_attributes.data." in new_key:
            new_key = new_key.replace(
                "model.model.node_attributes.data.",
                f"model.model.node_attributes.{dataset_name}."
            )
            rule_hits[f"node_attributes.data -> node_attributes.{dataset_name}"] += 1
        # --- bookkeeping ---
        if new_key == old_key:
            unchanged_keys.append(old_key)
        else:
            remapped_keys.append((old_key, new_key))

        updated_state_dict[new_key] = value

    # --- summary logging ---
    LOGGER.info("Key remapping summary:")
    LOGGER.info(f"  Total checkpoint keys: {len(checkpoint_state_dict)}")
    LOGGER.info(f"  Remapped keys: {len(remapped_keys)}")
    LOGGER.info(f"  Unchanged keys: {len(unchanged_keys)}")

    for rule, count in rule_hits.items():
        LOGGER.info(f"  {rule}: {count} keys")

    # --- examples for sanity checking ---
    LOGGER.info("Example remapped keys (up to 10):")
    for old_k, new_k in remapped_keys[:10]:
        LOGGER.info(f"  {old_k} -> {new_k}")

    # --- warn if something unexpected stays unmapped ---
    if unchanged_keys:
        LOGGER.info("Example unchanged keys (expected for dataset-agnostic modules):")
        for k in unchanged_keys[:10]:
            LOGGER.info(f"  {k}")

    LOGGER.info("Finished key remapping")
    return updated_state_dict


def transfer_learning_loading(model: torch.nn.Module, ckpt_path: Path | str) -> nn.Module:
    # Load the checkpoint
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=model.device)

    # apply chunking migration (fails silently otherwise leading to hard to debug issues)
    # this is due to loading with strict=False, planning to make this more robust in the future
    checkpoint = chunking_fix_migration(checkpoint)
    
    # Remap single-dataset checkpoint to multi-dataset model structure
    state_dict = checkpoint["state_dict"]
    state_dict = remap_single_to_multidataset(state_dict, dataset_name='era5', data_node_name="data_global")
    
    # Filter out layers with size mismatch
    model_state_dict = model.state_dict()
    for key in state_dict.copy():
        if key in model_state_dict and state_dict[key].shape != model_state_dict[key].shape:
            LOGGER.info("Skipping loading parameter: %s", key)
            LOGGER.info("Checkpoint shape: %s", str(state_dict[key].shape))
            LOGGER.info("Model shape: %s", str(model_state_dict[key].shape))

            del state_dict[key]  # Remove the mismatched key

    # Load the filtered st-ate_dict into the model
    model.load_state_dict(state_dict, strict=False)
    # Need name_to_index for data indices check
    # Handle both single-dataset and multi-dataset checkpoints
    try:
        # Try multi-dataset format first
        model._ckpt_model_name_to_index = checkpoint["hyper_parameters"]["data_indices"]["data"].name_to_index
    except (KeyError, AttributeError):
        # Fall back to single-dataset format
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
