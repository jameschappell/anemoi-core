from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from earthkit.regrid import db
from einops import rearrange
from torch.amp.autocast_mode import autocast
from anemoi.graphs.nodes.builders.from_reduced_gaussian import ReducedGaussianGridNodes


def get_grid_points(grid_type="o96", device=None):
    NodeBuilder = ReducedGaussianGridNodes(grid=grid_type, name="tmp")
    # coords shape: [N, 2] -> (lat, lon) in radians
    coords = NodeBuilder.get_coordinates()
    return torch.rad2deg(coords).to(device=device, dtype=torch.float32)


@dataclass
class RegridConfig:
    """Configuration for different grid types."""

    # Properties of output grid following regridding
    output_shape: tuple[int, int]  # (nlats, nlons) for output equiangular grid
    output_resolution: float  # degrees per grid point in output grid

    # Input grid type will be set from the dictionary key
    input_grid_type: str = ""  # e.g., "o96" or "n320"

    def __post_init__(self):
        # Eagerly populate at init time so FileLocks (anemoi.utils.caching and
        # earthkit.regrid.utils.caching) are acquired before training starts.
        self._input_grid_points = get_grid_points(self.input_grid_type).to(torch.float32)
        self._regridding_matrix = self._load_regridding_matrix()

    @property
    def output_nlats(self) -> int:
        return self.output_shape[0]

    @property
    def output_nlons(self) -> int:
        return self.output_shape[1]

    @property
    def input_grid_points(self) -> torch.Tensor:
        """Get input grid points for the reduced Gaussian grid.
        Returns torch tensor of shape [N, 2] where each row is [lat, lon] in degrees.
        """
        return self._input_grid_points

    @property
    def regridding_matrix(self) -> torch.Tensor:
        return self._regridding_matrix

    def get_regridding_matrix(self) -> torch.Tensor:
        """Only local rank 0 on each node downloads the matrix."""
        if dist.is_initialized():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))

            if local_rank == 0:
                regrid_matrix = self._load_regridding_matrix()

            dist.barrier()

            if local_rank != 0:
                regrid_matrix = self._load_regridding_matrix()

            return regrid_matrix
        return self._load_regridding_matrix()

    def _load_regridding_matrix(self) -> torch.Tensor:
        """Get the regridding matrix for transforming from input to output grid."""
        # url from which regridding matrices can be downloaded
        _SYSTEM_URL = "https://sites.ecmwf.int/repository/earthkit/regrid/db/1/"

        # setup grid specifications according to earthkit-regrid usage
        gridspec_in = {"grid": self.input_grid_type.upper()}
        gridspec_out = {"grid": [self.output_resolution, self.output_resolution]}

        # setup accessor and database from earthkit-regrid
        accessor = db.UrlAccessor(_SYSTEM_URL)
        matrix_database = db.MatrixDb(accessor)

        # find and extract the regridding matrix (uses linear interpolation)
        rm, _ = matrix_database.find(gridspec_in, gridspec_out, method="linear")

        # convert to a torch sparse COO tensor
        if rm is not None:
            rm_coo = rm.tocoo()
            indices = torch.from_numpy(np.vstack((rm_coo.row, rm_coo.col))).long()
            values = torch.tensor(rm_coo.data, dtype=torch.float32)
            shape = torch.Size(rm_coo.shape)
            return torch.sparse_coo_tensor(indices, values, shape)
        err_msg = f"Regridding matrix not found for {gridspec_in} to {gridspec_out}."
        raise ValueError(err_msg)

    def regrid_inference(
        self, tensor: torch.Tensor, regrid_matrix: torch.Tensor, requires_grad: bool = False
    ):
        """
        Regrid the input inference tensor from a reduced Gaussian grid to an equiangular grid using
        the precomputed regridding matrix.

        Args:
            tensor (torch.Tensor): shape [time, variables, ensemble, gridpoints].
            requires_grad (bool): Whether to compute gradients with respect to the input tensor.

        Returns:
            Regridded tensor of shape (time, n_vars, ensemble, nlats, nlons).
        """
        self.batch_size, self.n_vars, self.ensemble_size, self.n_gridpoints = tensor.shape

        # ensure tensors are on the same device
        if regrid_matrix.device != tensor.device:
            regrid_matrix = regrid_matrix.to(device=tensor.device)

        grad_context = torch.enable_grad() if requires_grad else torch.no_grad()
        with grad_context:
            # rearrange to shape (batch_size * ensemble * n_vars, n_gridpoints)
            tensor = rearrange(tensor, "bs var ens grid -> (bs ens var) grid")

            # self.regrid_matrix is a sparse COO matrix with shape (nlats*nlons, n_gridpoints)
            # autocast off allows sparse matrix multiplication
            # output tensor now has shape (batch_size, n_vars, ensemble, nlats, nlons)
            with autocast("cuda", enabled=False):
                return rearrange(
                    torch.sparse.mm(regrid_matrix, tensor.T).T,
                    "(bs ens var) (nlats nlons) -> bs var ens nlats nlons",
                    bs=self.batch_size,
                    ens=self.ensemble_size,
                    var=self.n_vars,
                    nlats=self.output_nlats,
                    nlons=self.output_nlons,
                )

    def regrid_training(
        self, tensor: torch.Tensor, regrid_matrix: torch.Tensor, requires_grad: bool = False
    ):
        """
        Regrid the input training tensor from a reduced Gaussian grid to an equiangular grid using
        the precomputed regridding matrix.

        Args:
            tensor (torch.Tensor): shape [batch_size, output_times, ensemble, n_gridpoints, n_vars].
            requires_grad (bool): Whether to compute gradients with respect to the input tensor.

        Returns:
            Regridded tensor of shape [batch_size * output_times * ensemble * n_vars, nlats, nlons] for training.
        """
        self.batch_size, self.output_times, self.ensemble_size, self.n_gridpoints, self.n_vars = (
            tensor.shape
        )

        # ensure tensors are on the same device
        if regrid_matrix.device != tensor.device:
            regrid_matrix = regrid_matrix.to(device=tensor.device)

        grad_context = torch.enable_grad() if requires_grad else torch.no_grad()
        with grad_context:
            # rearrange to shape (batch_size * output_times * ensemble * n_vars, n_gridpoints)
            tensor = rearrange(tensor, "bs ot ens grid var -> (bs ot ens var) grid")

            # self.regrid_matrix is a sparse COO matrix with shape (nlats*nlons, n_gridpoints)
            # autocast off allows sparse matrix multiplication
            # output tensor now has shape (batch_size * output_times * ensemble * n_vars, nlats, 
            # nlons) as expected by the spherical harmonic transform
            with autocast("cuda", enabled=False):
                return rearrange(
                    torch.sparse.mm(regrid_matrix, tensor.T).T,
                    "flat_bs (nlats nlons) -> flat_bs nlats nlons",
                    flat_bs=self.batch_size * self.output_times * self.ensemble_size * self.n_vars,
                    nlats=self.output_nlats,
                    nlons=self.output_nlons,
                )


def get_regrid_config(input_grid: str) -> RegridConfig:
    input_grid = input_grid.lower()
    if input_grid not in SUPPORTED_GRIDS:
        supported = ", ".join(SUPPORTED_GRIDS.keys())
        err_msg = f"Unsupported grid type: {input_grid}. Supported: {supported}"
        raise ValueError(err_msg)
    return SUPPORTED_GRIDS[input_grid]


SUPPORTED_GRIDS = {
    "n320": RegridConfig(output_shape=(721, 1440), output_resolution=0.25, input_grid_type="n320"),
    "o96": RegridConfig(output_shape=(181, 360), output_resolution=1.0, input_grid_type="o96"),
}
