# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import Iterable

import torch

from anemoi.models.interface import AnemoiModelInterface
from anemoi.models.preprocessing import StepwiseProcessors
from anemoi.training.losses.scalers.base_scaler import BaseUpdatingScaler
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)


class NaNMaskScaler(BaseUpdatingScaler):

    scale_dims: tuple[TensorDim] = (TensorDim.BATCH_SIZE, TensorDim.GRID, TensorDim.VARIABLE)

    def __init__(self, norm: str | None = None, use_processors_tendencies: bool = False, **kwargs) -> None:
        super().__init__(norm=norm)
        self.use_processors_tendencies = use_processors_tendencies
        del kwargs

    @staticmethod
    def _owned(t: torch.Tensor) -> torch.Tensor:
        if t.layout == torch.strided:
            return t.clone(memory_format=torch.contiguous_format)
        return t.clone()

    def _collect_processors(self, model: AnemoiModelInterface, dataset_name: str | None) -> list:
        processors = []

        if hasattr(model, "pre_processors"):
            assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."
            if dataset_name in model.pre_processors:
                processors.append(model.pre_processors[dataset_name])

        if self.use_processors_tendencies and hasattr(model, "pre_processors_tendencies"):
            assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."
            if dataset_name in model.pre_processors_tendencies:
                tendency_processors = model.pre_processors_tendencies[dataset_name]
                if isinstance(tendency_processors, StepwiseProcessors):
                    processors.extend(proc for proc in tendency_processors if proc is not None)
                else:
                    processors.append(tendency_processors)

        return processors

    def _iter_loss_masks(self, processors: Iterable) -> Iterable[torch.Tensor]:
        for pre_processors in processors:
            for pre_processor in pre_processors.processors.values():
                if not hasattr(pre_processor, "loss_mask_training"):
                    continue
                current_mask = pre_processor.loss_mask_training
                if current_mask is None or current_mask.numel() == 0:
                    continue
                yield self._owned(current_mask)

    @staticmethod
    def _combine_masks(masks: Iterable[torch.Tensor]) -> torch.Tensor | None:
        combined = None
        for current_mask in masks:
            combined = current_mask if combined is None else combined * current_mask
        return combined

    def on_batch_start(self, model: AnemoiModelInterface, dataset_name: str | None = None) -> torch.Tensor | None:
        processors = self._collect_processors(model=model, dataset_name=dataset_name)
        loss_weights_mask = self._combine_masks(self._iter_loss_masks(processors))
        return None if loss_weights_mask is None else self._owned(loss_weights_mask)
