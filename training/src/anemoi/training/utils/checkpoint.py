# (C) Copyright 2024-2026 Anemoi contributors.
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
from anemoi.training.train.methods.base import BaseTrainingModule
from anemoi.training.utils.variables_metadata import extract_variables_metadata_from_checkpoint
from anemoi.utils.checkpoints import save_metadata

chunking_fix_migration = importlib.import_module("anemoi.models.migrations.scripts.1762857428_chunking_fix").migrate
trainable_edge_perm_fix_migration = importlib.import_module(
    "anemoi.models.migrations.scripts.1779202136_trainable_edge_perm_fix",
).migrate

LOGGER = logging.getLogger(__name__)


def _filter_state_dict_size_mismatches(
    state_dict: dict[str, torch.Tensor],
    model_state_dict: dict[str, torch.Tensor],
) -> None:
    for key in list(state_dict):
        if key in model_state_dict and state_dict[key].shape != model_state_dict[key].shape:
            LOGGER.info("Skipping loading parameter: %s", key)
            LOGGER.info("Checkpoint shape: %s", str(state_dict[key].shape))
            LOGGER.info("Model shape: %s", str(model_state_dict[key].shape))

            del state_dict[key]


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
    module = BaseTrainingModule.load_from_checkpoint(lightning_checkpoint_path, weights_only=False)
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


def get_trainable_key(param_name: str) -> str | None:
    """Helper function used when transfer learning to identify changes in trainable_parameters numbers."""
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


def transfer_learning_loading(
    model: torch.nn.Module,
    ckpt_path: Path | str,
    model_config: dict,
    dataset_remapping: dict[str, str] | None = None,
) -> nn.Module:
    preview_limit = 20

    def add_preview(container: list[str], value: str) -> None:
        if len(container) < preview_limit:
            container.append(value)

    # Load the checkpoint
    # Load to CPU explictly, to avoid loading entire model on GPU initially
    # Modifications to the model occur on cpu,
    # The model will be sent to GPU when trainer.fit() is called
    LOGGER.debug("Loading checkpoint to device: cpu")
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu")

    # apply chunking migration (fails silently otherwise leading to hard to debug issues)
    # this is due to loading with strict=False, planning to make this more robust in the future
    checkpoint = chunking_fix_migration(checkpoint)

    # extract trainable_parameters dictionary from the model config
    trainable_parameters = model_config.trainable_parameters

    # Refresh processor stats from the current dataset if configured.
    model._update_checkpoint_state_dict_for_load(checkpoint)

    # check whether sizes of components are compatible, either matching or differing by
    # trainable_parameters
    state_dict = checkpoint["state_dict"]
    _filter_state_dict_size_mismatches(state_dict, model.state_dict())

    # Runtime migration: the graph-provider permutation depends on instantiated provider state.
    checkpoint = trainable_edge_perm_fix_migration(checkpoint, model)
    state_dict = checkpoint["state_dict"]
    checkpoint_param_count = len(state_dict)

    exact_match_count = 0
    partial_load_count = 0

    skipped_missing_target_count = 0
    skipped_different_ndim_count = 0
    skipped_missing_growth_key_count = 0
    skipped_missing_allowed_growth_count = 0
    skipped_shape_mismatch_count = 0

    partial_loaded_preview: list[str] = []
    skipped_missing_target_preview: list[str] = []
    skipped_different_ndim_preview: list[str] = []
    skipped_missing_growth_key_preview: list[str] = []
    skipped_missing_allowed_growth_preview: list[str] = []
    skipped_shape_mismatch_preview: list[str] = []

    # Remap dataset names in state_dict before loading
    if dataset_remapping:
        LOGGER.info("Applying dataset remapping: %s", dataset_remapping)
        state_dict = remap_checkpoint_dataset(state_dict, dataset_remapping)

    model_state_dict = model.state_dict()

    for key in list(state_dict.keys()):
        if key not in model_state_dict:
            skipped_missing_target_count += 1
            add_preview(skipped_missing_target_preview, key)
            continue

        ckpt_tensor = state_dict[key]
        model_tensor = model_state_dict[key]

        if ckpt_tensor.shape == model_tensor.shape:
            exact_match_count += 1
            continue  # perfect match

        if ckpt_tensor.ndim != model_tensor.ndim:
            skipped_different_ndim_count += 1
            add_preview(skipped_different_ndim_preview, key)
            LOGGER.info("Skipping %s (different ndim)", key)
            del state_dict[key]
            continue

        # check whether the size of the parameter grows by the number of trainable parameters
        # if so, load it into the matching slice of the tensor
        growth_key = get_trainable_key(key)

        if growth_key is None:
            skipped_missing_growth_key_count += 1
            add_preview(skipped_missing_growth_key_preview, key)
            LOGGER.info("Skipping %s (no matching trainable parameter growth key)", key)
            del state_dict[key]
            continue

        allowed_growth = trainable_parameters.get(growth_key, None)

        if allowed_growth is None:
            skipped_missing_allowed_growth_count += 1
            add_preview(skipped_missing_allowed_growth_preview, key)
            LOGGER.info("Skipping %s (growth key '%s' not configured)", key, growth_key)
            del state_dict[key]
            continue

        # compute per-dimension differences
        diffs = [m - c for c, m in zip(ckpt_tensor.shape, model_tensor.shape, strict=False)]

        # only allow change in parameter size in ONE dimension equal to allowed_growth
        # if checkpoint parameter has shape [num_channels, size], model has [num_channels, size + allowed_growth]
        # then can load weights into first [num_channels, size] of the model weights
        # i.e. only the trainable_parameters are initialised from scratch
        positive_diffs = [d for d in diffs if d > 0]

        if positive_diffs == [allowed_growth] and all(d >= 0 for d in diffs):
            partial_load_count += 1
            add_preview(partial_loaded_preview, key)
            LOGGER.info("Partially loading %s with allowed growth %d from key %s", key, allowed_growth, growth_key)
            LOGGER.info("Checkpoint shape: %s", tuple(ckpt_tensor.shape))
            LOGGER.info("Model shape: %s", tuple(model_tensor.shape))

            new_tensor = model_tensor.clone()
            slices = tuple(slice(0, min(c, m)) for c, m in zip(ckpt_tensor.shape, model_tensor.shape, strict=False))
            new_tensor[slices] = ckpt_tensor[slices]
            state_dict[key] = new_tensor
        else:
            skipped_shape_mismatch_count += 1
            add_preview(skipped_shape_mismatch_preview, key)
            LOGGER.info("Skipping %s (shape change not matching config)", key)
            LOGGER.info("Checkpoint shape: %s", tuple(ckpt_tensor.shape))
            LOGGER.info("Model shape: %s", tuple(model_tensor.shape))
            del state_dict[key]

    # Load the filtered st-ate_dict into the model
    load_result = model.load_state_dict(state_dict, strict=False)

    missing_keys = list(getattr(load_result, "missing_keys", []))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
    missing_preview = missing_keys[:preview_limit]
    unexpected_preview = unexpected_keys[:preview_limit]

    LOGGER.info(
        "Transfer learning load summary from %s: checkpoint_params=%d, exact_matches=%d, partial_matches=%d, "
        "skipped_missing_target=%d, skipped_different_ndim=%d, skipped_missing_growth_key=%d, "
        "skipped_missing_allowed_growth=%d, skipped_shape_mismatch=%d",
        ckpt_path,
        checkpoint_param_count,
        exact_match_count,
        partial_load_count,
        skipped_missing_target_count,
        skipped_different_ndim_count,
        skipped_missing_growth_key_count,
        skipped_missing_allowed_growth_count,
        skipped_shape_mismatch_count,
    )

    if partial_loaded_preview:
        LOGGER.info("Partially loaded parameters (first %d): %s", len(partial_loaded_preview), partial_loaded_preview)
    if skipped_missing_target_preview:
        LOGGER.info(
            "Skipped (missing in target model) parameters (first %d): %s",
            len(skipped_missing_target_preview),
            skipped_missing_target_preview,
        )
    if skipped_different_ndim_preview:
        LOGGER.info(
            "Skipped (different ndim) parameters (first %d): %s",
            len(skipped_different_ndim_preview),
            skipped_different_ndim_preview,
        )
    if skipped_missing_growth_key_preview:
        LOGGER.info(
            "Skipped (no growth key rule) parameters (first %d): %s",
            len(skipped_missing_growth_key_preview),
            skipped_missing_growth_key_preview,
        )
    if skipped_missing_allowed_growth_preview:
        LOGGER.info(
            "Skipped (growth key missing in config) parameters (first %d): %s",
            len(skipped_missing_allowed_growth_preview),
            skipped_missing_allowed_growth_preview,
        )
    if skipped_shape_mismatch_preview:
        LOGGER.info(
            "Skipped (shape mismatch) parameters (first %d): %s",
            len(skipped_shape_mismatch_preview),
            skipped_shape_mismatch_preview,
        )

    LOGGER.info(
        "Model load_state_dict results: missing_keys=%d, unexpected_keys=%d",
        len(missing_keys),
        len(unexpected_keys),
    )
    if missing_preview:
        LOGGER.info("Missing keys reported by load_state_dict (first %d): %s", len(missing_preview), missing_preview)
    if unexpected_preview:
        LOGGER.info(
            "Unexpected keys reported by load_state_dict (first %d): %s",
            len(unexpected_preview),
            unexpected_preview,
        )

    # Needed for data indices check - data_indices is a dict[str, IndexCollection]
    data_indices = checkpoint["hyper_parameters"]["data_indices"]
    if isinstance(data_indices, dict):
        model._ckpt_model_name_to_index = {
            k: v.name_to_index for k, v in data_indices.items() if hasattr(v, "name_to_index")
        }
    else:
        # Old format: data_indices is a single IndexCollection object (not dict)
        msg = (
            f"Checkpoint at '{ckpt_path}' was created with an older version of anemoi-core "
            "that does not support multi-dataset training. This checkpoint is incompatible "
            "with transfer learning in the current version."
        )
        raise TypeError(msg)

    # Extract variables_metadata for unit compatibility check
    model._ckpt_variables_metadata = extract_variables_metadata_from_checkpoint(
        checkpoint,
        model._ckpt_model_name_to_index,
    )

    return model


def freeze_submodule_by_name(module: nn.Module, target_name: str) -> dict[str, Any]:
    """Recursively freezes the parameters of a submodule with the specified name.

    Parameters
    ----------
    module : torch.nn.Module
        Pytorch model
    target_name : str
        The name of the submodule to freeze.

    Returns
    -------
    dict[str, Any]
        Summary of freeze operations, including matched module paths and
        frozen parameter names.
    """
    preview_limit = 20
    matched_module_paths: list[str] = []
    frozen_parameter_names: list[str] = []
    frozen_parameter_count = 0
    frozen_parameter_elements = 0
    visited_param_ids: set[int] = set()

    def _freeze_submodule(current_module: nn.Module, module_path: str) -> None:
        nonlocal frozen_parameter_count
        nonlocal frozen_parameter_elements

        for child_name, child_module in current_module.named_children():
            child_path = f"{module_path}.{child_name}" if module_path else child_name

            if child_name == target_name:
                matched_module_paths.append(child_path)

                for param_name, param in child_module.named_parameters(recurse=True):
                    param_id = id(param)
                    if param_id in visited_param_ids:
                        continue

                    visited_param_ids.add(param_id)
                    param.requires_grad = False
                    frozen_parameter_count += 1
                    frozen_parameter_elements += int(param.numel())

                    if len(frozen_parameter_names) < preview_limit:
                        full_param_name = f"{child_path}.{param_name}" if param_name else child_path
                        frozen_parameter_names.append(full_param_name)
            else:
                _freeze_submodule(child_module, child_path)

    _freeze_submodule(module, "")

    if matched_module_paths:
        LOGGER.info(
            "Froze %d parameter tensors (%d total elements) across %d submodule(s) named '%s'",
            frozen_parameter_count,
            frozen_parameter_elements,
            len(matched_module_paths),
            target_name,
        )
        LOGGER.info(
            "Matched submodule paths (first %d): %s",
            min(preview_limit, len(matched_module_paths)),
            matched_module_paths[:preview_limit],
        )
        LOGGER.info(
            "Frozen parameter names (first %d): %s",
            len(frozen_parameter_names),
            frozen_parameter_names,
        )
    else:
        LOGGER.warning("No submodule named '%s' found to freeze", target_name)

    return {
        "target_name": target_name,
        "matched_module_paths": matched_module_paths,
        "frozen_parameter_count": frozen_parameter_count,
        "frozen_parameter_elements": frozen_parameter_elements,
        "frozen_parameter_preview": frozen_parameter_names,
    }


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
