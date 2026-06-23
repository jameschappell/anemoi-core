# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Concrete loading strategy implementations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from anemoi.training.checkpoint.exceptions import CheckpointLoadError
from anemoi.training.checkpoint.loading.base import LoadingStrategy

if TYPE_CHECKING:
    from anemoi.training.checkpoint.base import CheckpointContext

LOGGER = logging.getLogger(__name__)


class WeightsOnlyLoader(LoadingStrategy):
    """Load only model weights, discarding optimizer and scheduler state.

    This is the simplest loading strategy: extract the state dict from
    checkpoint data, load it into the model, and explicitly discard any
    optimizer/scheduler state. Useful for cold-start scenarios where you
    want pretrained weights but a fresh optimizer.

    Parameters
    ----------
    strict : bool, optional
        Whether to require an exact match between checkpoint keys and
        model keys (default: False)
    """

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        """Load weights into model, discard optimizer/scheduler.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context with ``checkpoint_data`` and ``model`` set.

        Returns
        -------
        CheckpointContext
            Context with weights loaded and optimizer/scheduler cleared.
        """
        state_dict = self._extract_state_dict(context)

        try:
            context.model.load_state_dict(state_dict, strict=self.strict)
        except RuntimeError as e:
            msg = f"Failed to load state dict into model: {e}"
            raise CheckpointLoadError(msg) from e

        self._preserve_anemoi_metadata(context.model, context.checkpoint_data)
        self._mark_weights_loaded(context.model)

        # Discard optimizer/scheduler — weights-only means fresh training state
        context.optimizer = None
        context.scheduler = None

        context.metadata["loading_strategy"] = "weights_only"

        LOGGER.info("Loaded weights only (strict=%s), optimizer/scheduler discarded", self.strict)

        return context


class TransferLearningLoader(LoadingStrategy):
    """Flexible loading for transfer learning scenarios.

    Filters the source state dict to only include keys compatible with the
    target model (matching key names and tensor shapes), then loads the
    filtered weights. Keys that are missing in the target or have shape
    mismatches are skipped rather than raising an error.

    The filter is non-mutating: it builds a new dict and never modifies the
    original ``checkpoint_data["state_dict"]``.

    Parameters
    ----------
    skip_mismatched : bool, optional
        Whether to skip keys with mismatched shapes (default: True).
        If False, shape mismatches raise ``CheckpointIncompatibleError``.
    remap_dataset : dict[str, str] | None, optional
        Optional checkpoint dataset-name remapping applied to state-dict
        keys before filtering, e.g. ``{'era5': 'gm'}``.
    exclude_key_prefixes : list[str] | None, optional
        Optional list of key prefixes to always exclude from transfer.
        This is useful for dataset-dependent tensors (e.g. preprocessors,
        postprocessors, node attributes) that should be kept from the new model.
    """

    def __init__(
        self,
        skip_mismatched: bool = True,
        remap_dataset: dict[str, str] | None = None,
        exclude_key_prefixes: list[str] | None = None,
    ) -> None:
        self.skip_mismatched = skip_mismatched
        self.remap_dataset = remap_dataset or {}
        self.exclude_key_prefixes = tuple(exclude_key_prefixes or [])

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        """Filter and load compatible weights from checkpoint.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context with ``checkpoint_data`` and ``model`` set.

        Returns
        -------
        CheckpointContext
            Context with compatible weights loaded and metadata updated.
        """
        from anemoi.training.checkpoint.loading.utils import filter_state_dict
        from anemoi.training.checkpoint.loading.utils import remap_dataset_keys

        source_state = self._extract_state_dict(context)
        target_state = context.model.state_dict()

        if self.remap_dataset:
            source_state, remapped_keys, collisions = remap_dataset_keys(
                source_state,
                self.remap_dataset,
                target_keys=set(target_state.keys()),
            )
            context.metadata["dataset_remap"] = dict(self.remap_dataset)
            context.metadata["dataset_remap_key_count"] = len(remapped_keys)
            context.metadata["dataset_remap_collision_count"] = sum(len(v) for v in collisions.values())

            LOGGER.info(
                "Transfer learning dataset remap applied: %s (remapped %d keys, collisions %d)",
                self.remap_dataset,
                len(remapped_keys),
                sum(len(v) for v in collisions.values()),
            )

            preview_limit = 20
            remapped_preview = list(remapped_keys.items())[:preview_limit]
            if remapped_preview:
                LOGGER.info(
                    "Transfer learning remapped parameter keys (first %d): %s",
                    len(remapped_preview),
                    remapped_preview,
                )

            collision_preview = list(collisions.items())[:preview_limit]
            if collision_preview:
                LOGGER.warning(
                    "Transfer learning remap collisions (first %d): %s",
                    len(collision_preview),
                    collision_preview,
                )

        excluded_keys: list[str] = []
        if self.exclude_key_prefixes:
            filtered_source_state = {}
            for key, value in source_state.items():
                if key.startswith(self.exclude_key_prefixes):
                    excluded_keys.append(key)
                else:
                    filtered_source_state[key] = value
            source_state = filtered_source_state

            context.metadata["exclude_key_prefixes"] = list(self.exclude_key_prefixes)
            context.metadata["excluded_param_count"] = len(excluded_keys)

            LOGGER.info(
                "Transfer learning excluded %d keys by prefix filters: %s",
                len(excluded_keys),
                list(self.exclude_key_prefixes),
            )

            preview_limit = 20
            excluded_preview = excluded_keys[:preview_limit]
            if excluded_preview:
                LOGGER.info(
                    "Transfer learning excluded parameter keys (first %d): %s",
                    len(excluded_preview),
                    excluded_preview,
                )

        filtered, skipped = filter_state_dict(source_state, target_state)

        for key in excluded_keys:
            skipped[key] = "Excluded by prefix filter"

        missing_target_keys = [k for k, reason in skipped.items() if reason == "Key not in target"]
        shape_mismatch_keys = [k for k, reason in skipped.items() if reason.startswith("Shape mismatch")]
        excluded_by_filter_keys = [k for k, reason in skipped.items() if reason == "Excluded by prefix filter"]

        shape_mismatch_details = []
        for key in shape_mismatch_keys:
            source_tensor = source_state.get(key)
            target_tensor = target_state.get(key)

            source_shape = tuple(source_tensor.shape) if source_tensor is not None and hasattr(source_tensor, "shape") else None
            target_shape = tuple(target_tensor.shape) if target_tensor is not None and hasattr(target_tensor, "shape") else None

            if source_shape is not None and target_shape is not None and len(source_shape) == len(target_shape):
                shape_delta = tuple(target_dim - source_dim for source_dim, target_dim in zip(source_shape, target_shape))
            else:
                shape_delta = None

            shape_mismatch_details.append(
                {
                    "key": key,
                    "source_shape": source_shape,
                    "target_shape": target_shape,
                    "shape_delta": shape_delta,
                },
            )

        if not self.skip_mismatched:
            shape_skipped = {k: v for k, v in skipped.items() if "Shape mismatch" in v}
            if shape_skipped:
                from anemoi.training.checkpoint.exceptions import CheckpointIncompatibleError

                msg = f"Shape mismatches found and skip_mismatched=False: {shape_skipped}"
                raise CheckpointIncompatibleError(msg)

        try:
            context.model.load_state_dict(filtered, strict=False)
        except RuntimeError as e:
            msg = f"Failed to load filtered state dict into model: {e}"
            raise CheckpointLoadError(msg) from e

        # NOTE: Do NOT call _preserve_anemoi_metadata here.
        # For transfer learning, we intentionally discard the checkpoint's variable
        # mapping (data_indices) because we're loading into a model with different
        # architecture/variables. The sanity check callback will fall back to using
        # the new dataset's indices, which is the correct behavior.
        self._mark_weights_loaded(context.model)

        # Discard optimizer/scheduler — transfer learning means fresh training state
        context.optimizer = None
        context.scheduler = None

        context.metadata["loading_strategy"] = "transfer_learning"
        context.metadata["transferred_params"] = list(filtered.keys())
        context.metadata["skipped_params"] = skipped
        context.metadata["transferred_param_count"] = len(filtered)
        context.metadata["skipped_param_count"] = len(skipped)
        context.metadata["skipped_missing_target_count"] = len(missing_target_keys)
        context.metadata["skipped_shape_mismatch_count"] = len(shape_mismatch_keys)
        context.metadata["skipped_shape_mismatch_details"] = shape_mismatch_details
        context.metadata["skipped_excluded_by_filter_count"] = len(excluded_by_filter_keys)

        LOGGER.info(
            "Transfer learning: loaded %d params, skipped %d (missing target: %d, shape mismatch: %d, excluded: %d)",
            len(filtered),
            len(skipped),
            len(missing_target_keys),
            len(shape_mismatch_keys),
            len(excluded_by_filter_keys),
        )

        preview_limit = 20
        transferred_preview = list(filtered.keys())[:preview_limit]
        skipped_missing_preview = missing_target_keys[:preview_limit]
        skipped_shape_preview = shape_mismatch_keys[:preview_limit]
        skipped_excluded_preview = excluded_by_filter_keys[:preview_limit]

        if transferred_preview:
            LOGGER.info(
                "Transfer learning loaded parameter names (first %d): %s",
                len(transferred_preview),
                transferred_preview,
            )
        if skipped_missing_preview:
            LOGGER.info(
                "Transfer learning skipped (missing target key) parameter names (first %d): %s",
                len(skipped_missing_preview),
                skipped_missing_preview,
            )
        if skipped_shape_preview:
            LOGGER.info(
                "Transfer learning skipped (shape mismatch) parameter names (first %d): %s",
                len(skipped_shape_preview),
                skipped_shape_preview,
            )

            shape_detail_preview = [
                (
                    detail["key"],
                    detail["source_shape"],
                    detail["target_shape"],
                    detail["shape_delta"],
                )
                for detail in shape_mismatch_details[:preview_limit]
            ]
            LOGGER.info(
                "Transfer learning shape mismatch details (first %d) as "
                "(key, checkpoint_shape, target_shape, target-minus-checkpoint): %s",
                len(shape_detail_preview),
                shape_detail_preview,
            )
        if skipped_excluded_preview:
            LOGGER.info(
                "Transfer learning skipped (excluded by prefix filter) parameter names (first %d): %s",
                len(skipped_excluded_preview),
                skipped_excluded_preview,
            )

        return context


class WarmStartLoader(LoadingStrategy):
    """Resume training with full state restoration.

    Restores model weights, optimizer state, scheduler state, and
    training progress (epoch, global_step). This is the strategy to use
    when resuming an interrupted training run on the same architecture.

    Unlike other strategies, WarmStart uses ``strict=True`` for model
    weights because an exact architecture match is expected when resuming.
    Optimizer and scheduler states are restored from Lightning-format
    checkpoint keys (``optimizer_states``, ``lr_schedulers``).
    """

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        """Restore full training state from checkpoint.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context with ``checkpoint_data``, ``model``, and
            optionally ``optimizer`` and ``scheduler`` set.

        Returns
        -------
        CheckpointContext
            Context with all training state restored.
        """
        from anemoi.training.checkpoint.exceptions import CheckpointIncompatibleError

        # 1. Model weights (strict — exact match expected for resume)
        state_dict = self._extract_state_dict(context)
        try:
            context.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            msg = f"WarmStart requires exact model match: {e}"
            raise CheckpointIncompatibleError(msg) from e

        # 2. Optimizer state
        if context.optimizer is not None and "optimizer_states" in context.checkpoint_data:
            context.optimizer.load_state_dict(context.checkpoint_data["optimizer_states"][0])
        elif context.optimizer is not None:
            LOGGER.warning("Checkpoint has no 'optimizer_states'; optimizer state not restored")

        # 3. Scheduler state
        if context.scheduler is not None and "lr_schedulers" in context.checkpoint_data:
            context.scheduler.load_state_dict(context.checkpoint_data["lr_schedulers"][0])
        elif context.scheduler is not None:
            LOGGER.warning("Checkpoint has no 'lr_schedulers'; scheduler state not restored")

        # 4. Training progress
        context.metadata["epoch"] = context.checkpoint_data.get("epoch", 0)
        context.metadata["global_step"] = context.checkpoint_data.get("global_step", 0)

        # 5. Anemoi metadata
        self._preserve_anemoi_metadata(context.model, context.checkpoint_data)
        self._mark_weights_loaded(context.model)
        context.metadata["loading_strategy"] = "warm_start"

        LOGGER.info(
            "Warm start: restored full state (epoch=%d, global_step=%d)",
            context.metadata["epoch"],
            context.metadata["global_step"],
        )

        return context


class ColdStartLoader(WeightsOnlyLoader):
    """Start fresh training from pretrained weights.

    Loads model weights (via WeightsOnlyLoader), then explicitly resets
    training state (epoch, global_step) to zero and records the
    pretrained checkpoint source. Optimizer and scheduler are discarded.
    """

    async def process(self, context: CheckpointContext) -> CheckpointContext:
        """Load weights and reset training state to zero.

        Parameters
        ----------
        context : CheckpointContext
            Pipeline context with ``checkpoint_data`` and ``model`` set.

        Returns
        -------
        CheckpointContext
            Context with weights loaded and training state reset.
        """
        context = await super().process(context)

        context.metadata["epoch"] = 0
        context.metadata["global_step"] = 0
        context.metadata["loading_strategy"] = "cold_start"
        context.metadata["pretrained_from"] = str(context.checkpoint_path) if context.checkpoint_path else None

        LOGGER.info("Cold start: training state reset, pretrained from %s", context.checkpoint_path)

        return context
