# (C) Copyright 2026 Anemoi contributors.

from __future__ import annotations

import logging
from typing import Any
from typing import Optional

import numpy as np
import torch
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing import ForwardOnlyPreProcessor

LOGGER = logging.getLogger(__name__)


class MatrixRegridder(ForwardOnlyPreProcessor):
    """Forward-only sparse-matrix regridding over the grid axis.

    Expected input shape: [..., grid, vars]
    Output shape:         [..., new_grid, vars]

    Optional config keys for startup validation:
      - source_nodes: {node_builder: {_target_: ..., ...}} or {_target_: ..., ...}
      - target_nodes: {node_builder: {_target_: ..., ...}} or {_target_: ..., ...}
      - source_num_nodes: int  # optional direct override
      - target_num_nodes: int  # optional direct override
    """

    def __init__(
        self,
        config=None,
        data_indices: Optional[IndexCollection] = None,
        statistics: Optional[dict] = None,
    ) -> None:
        base_cfg = {"default": "none"}
        super().__init__(config=base_cfg, data_indices=data_indices, statistics=statistics)

        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        config = dict(config or {})

        matrix_path = config.get("matrix_path")
        if not matrix_path:
            raise ValueError("MatrixRegridder requires config.matrix_path")

        self.matrix_path = str(matrix_path)
        self.nan_safe = bool(config.get("nan_safe", False))
        self.min_valid_weight = float(config.get("min_valid_weight", 1e-12))

        self.regrid_matrix = self._load_matrix(self.matrix_path)
        self.source_grid_size = int(self.regrid_matrix.shape[1])
        self.target_grid_size = int(self.regrid_matrix.shape[0])

        # Useful for future sharding logic that needs to detect grid-changing preprocessors.
        self.changes_grid_size = True

        LOGGER.info(
            "MatrixRegridder loaded matrix %s with shape (target=%d, source=%d)",
            self.matrix_path,
            self.target_grid_size,
            self.source_grid_size,
        )

        self._validate_matrix_shape_against_configured_nodes(config)

    @staticmethod
    def _as_plain(value: Any) -> Any:
        if isinstance(value, DictConfig):
            return OmegaConf.to_container(value, resolve=True)
        return value

    @staticmethod
    def _load_matrix(path: str) -> torch.Tensor:
        with np.load(path, allow_pickle=False) as loaded:
            shape = tuple(np.asarray(loaded["matrix_shape"], dtype=np.int64).tolist())
            crow_indices = torch.from_numpy(np.asarray(loaded["matrix_indptr"], dtype=np.int64))
            col_indices = torch.from_numpy(np.asarray(loaded["matrix_indices"], dtype=np.int64))
            values = torch.from_numpy(np.asarray(loaded["matrix_data"], dtype=np.float32))

        return torch.sparse_csr_tensor(
            crow_indices,
            col_indices,
            values,
            size=shape,
        )

    def _extract_node_builder_cfg(self, node_spec: Any, role: str) -> dict | None:
        if node_spec is None:
            return None

        node_spec = self._as_plain(node_spec)
        if not isinstance(node_spec, dict):
            raise TypeError(f"{role}_nodes must be a mapping, got {type(node_spec).__name__}")

        # Accept graph-style wrappers:
        #   source_nodes:
        #     node_builder:
        #       _target_: ...
        builder_cfg = node_spec.get("node_builder", node_spec)
        builder_cfg = self._as_plain(builder_cfg)

        if not isinstance(builder_cfg, dict) or "_target_" not in builder_cfg:
            raise ValueError(f"{role}_nodes must contain a node builder config with _target_. Got: {builder_cfg}")

        return dict(builder_cfg)

    def _resolve_num_nodes_from_builder(self, node_spec: Any, role: str) -> int | None:
        builder_cfg = self._extract_node_builder_cfg(node_spec, role)
        if builder_cfg is None:
            return None

        builder_name = builder_cfg.pop("name", f"regrid_{role}_nodes")

        try:
            builder = instantiate(builder_cfg, name=builder_name)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to instantiate {role} node builder for MatrixRegridder: {builder_cfg}") from exc

        if not hasattr(builder, "get_coordinates"):
            raise TypeError(f"{role} node builder {type(builder).__name__} has no get_coordinates() method")

        try:
            coords = builder.get_coordinates()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to read coordinates from {role} node builder {type(builder).__name__}") from exc

        if isinstance(coords, torch.Tensor):
            if coords.ndim < 1:
                raise ValueError(f"{role} node coordinates tensor must be at least 1D, got {tuple(coords.shape)}")
            n_nodes = int(coords.shape[0])
        else:
            coords_arr = np.asarray(coords)
            if coords_arr.ndim < 1:
                raise ValueError(f"{role} node coordinates array must be at least 1D, got shape {coords_arr.shape}")
            n_nodes = int(coords_arr.shape[0])

        if n_nodes <= 0:
            raise ValueError(f"{role} node builder returned non-positive node count: {n_nodes}")

        LOGGER.info(
            "MatrixRegridder %s node builder %s resolved %d nodes",
            role,
            builder_cfg.get("_target_"),
            n_nodes,
        )
        return n_nodes

    def _validate_matrix_shape_against_configured_nodes(self, config: dict) -> None:
        # Optional direct overrides.
        source_n = config.get("source_num_nodes")
        target_n = config.get("target_num_nodes")

        # Optional node-builder based resolution.
        if source_n is None and "source_nodes" in config:
            source_n = self._resolve_num_nodes_from_builder(config.get("source_nodes"), role="source")
        if target_n is None and "target_nodes" in config:
            target_n = self._resolve_num_nodes_from_builder(config.get("target_nodes"), role="target")

        errors = []

        if source_n is not None:
            source_n = int(source_n)
            if source_n != self.source_grid_size:
                errors.append(f"source nodes={source_n} does not match matrix source dimension={self.source_grid_size}")

        if target_n is not None:
            target_n = int(target_n)
            if target_n != self.target_grid_size:
                errors.append(f"target nodes={target_n} does not match matrix target dimension={self.target_grid_size}")

        if errors:
            details = "\n  - ".join(errors)
            raise ValueError(
                "MatrixRegridder node validation failed:\n" f"  - {details}\n" f"  - matrix_path={self.matrix_path}"
            )

        if source_n is not None or target_n is not None:
            LOGGER.info(
                "MatrixRegridder node validation passed: target=%d, source=%d",
                self.target_grid_size,
                self.source_grid_size,
            )

    def _matrix_for(self, x2d: torch.Tensor) -> torch.Tensor:
        matrix = self.regrid_matrix
        if matrix.device != x2d.device or matrix.dtype != x2d.dtype:
            matrix = matrix.to(device=x2d.device, dtype=x2d.dtype)
            self.regrid_matrix = matrix
        return matrix

    @staticmethod
    def _left_sparse_mm(matrix: torch.Tensor, x2d: torch.Tensor) -> torch.Tensor:
        # matrix: [n_target, n_source], x2d: [N, n_source] -> [N, n_target]
        return torch.sparse.mm(matrix, x2d.transpose(0, 1)).transpose(0, 1)

    def _apply_matrix(self, x2d: torch.Tensor) -> torch.Tensor:
        matrix = self._matrix_for(x2d)

        if not self.nan_safe:
            return self._left_sparse_mm(matrix, x2d)

        invalid = ~torch.isfinite(x2d)
        if not bool(invalid.any()):
            return self._left_sparse_mm(matrix, x2d)

        # Lower-allocation NaN-safe path:
        # clone + masked_fill_ avoids extra zeros_like/where temporaries.
        x_filled = x2d.clone()
        x_filled.masked_fill_(invalid, 0.0)

        out = self._left_sparse_mm(matrix, x_filled)
        invalid_weight = self._left_sparse_mm(matrix, invalid.to(dtype=x2d.dtype))
        out.masked_fill_(invalid_weight > self.min_valid_weight, torch.nan)
        return out

    def transform(self, x: torch.Tensor, in_place: bool = True, **kwargs) -> torch.Tensor:
        del kwargs
        if not in_place:
            x = x.clone()

        if x.ndim < 2:
            raise ValueError(f"Expected at least 2 dims [..., grid, vars], got shape {tuple(x.shape)}")

        n_source = x.shape[-2]
        if n_source != self.source_grid_size:
            raise ValueError(
                f"Grid size mismatch: tensor has {n_source} points, matrix expects {self.source_grid_size}"
            )

        n_vars = x.shape[-1]
        leading_shape = x.shape[:-2]
        n_leading = int(np.prod(leading_shape, dtype=np.int64)) if leading_shape else 1

        # [..., grid, vars] -> [(leading * vars), grid]
        x2d = rearrange(x, "... grid vars -> (... vars) grid").contiguous()
        y2d = self._apply_matrix(x2d)

        # [(leading * vars), new_grid] -> [leading, new_grid, vars] -> [..., new_grid, vars]
        y = rearrange(y2d, "(lead vars) new_grid -> lead new_grid vars", lead=n_leading, vars=n_vars)
        return y.reshape(*leading_shape, self.target_grid_size, n_vars)
