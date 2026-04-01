import logging
import re

import numpy as np
import torch
from requests.exceptions import HTTPError

from anemoi.graphs.generate.masks import KNNAreaMaskBuilder
from anemoi.graphs.generate.utils import get_coordinates_ordering
from anemoi.utils.grids import grids

LOGGER = logging.getLogger(__name__)


def get_latlon_coords_gaussian(grid: str) -> np.ndarray:
    """Get the latitude and longitude coordinates (in radians) of a reduced gaussian grid.

    Parameters
    ----------
    grid : str
        The reduced gaussian grid identifier, e.g. 'O96', 'N320'.
        If the grid is not found in the registry and starts with 'O',
        falls back to generating it locally via reduced_gaussian_gridpoints().

    Returns
    -------
    np.ndarray of shape (num_nodes, 2)
        The latitude and longitude coordinates, in radians.
    """
    try:
        grid_data = grids(grid)
        lats = np.deg2rad(grid_data["latitudes"])
        lons = np.deg2rad(grid_data["longitudes"])
    except HTTPError:
        if not re.match(r"^[Oo]\d+$", grid):
            raise ValueError(f"Grid '{grid}' not found in registry and does not match expected format 'O{{n_points}}'.")
        n_points = int(re.match(r"^[Oo](\d+)$", grid).group(1))
        LOGGER.warning(
            "Grid '%s' not found in registry. Falling back to  octahedral_reduced_gaussian_gridpoints(n_points=%d).",
            grid,
            n_points,
        )
        lats_deg, lons_deg = octahedral_reduced_gaussian_gridpoints(n_points=n_points)
        lats = np.deg2rad(lats_deg)
        lons = np.deg2rad(lons_deg)

    return np.stack([lats, lons], axis=-1)


def octahedral_reduced_gaussian_gridpoints(n_points=96, dtype=np.float64):
    """
    Generate coordinates for the ECMWF octahedral reduced Gaussian grid.
    """
    N = n_points * 2

    # Gaussian latitudes (north -> south)
    x, _ = np.polynomial.legendre.leggauss(N)
    gauss_lats = np.degrees(np.arcsin(x))[::-1].astype(dtype, copy=False)

    # Number of longitudes per latitude (octahedral)
    nlons_half = 16 + 4 * np.arange(1, N // 2 + 1)
    nlons = np.concatenate([nlons_half, nlons_half[::-1]]).astype(np.int64, copy=False)

    # Vectorized full coordinate arrays
    lats = np.repeat(gauss_lats, nlons)

    n_total = int(nlons.sum())
    starts = np.cumsum(np.r_[0, nlons[:-1]])      # start index per latitude ring
    idx_in_ring = np.arange(n_total) - np.repeat(starts, nlons)
    nlon_per_point = np.repeat(nlons, nlons)

    lons = (idx_in_ring * (360.0 / nlon_per_point)).astype(dtype, copy=False)
    return lats, lons


def create_stretched_reduced_gaussian_nodes(
    global_grid: str,
    lam_grid: str,
    area_mask_builder: KNNAreaMaskBuilder,
) -> torch.Tensor:
    """Creates nodes from two reduced gaussian grids with different resolutions.

    The global_grid is used to define the nodes outside the Area Of Interest (AOI),
    while the lam_grid is used to define the nodes inside the AOI.

    Parameters
    ----------
    global_grid : str
        Global (coarser) reduced gaussian grid identifier, e.g. 'O96'.
    lam_grid : str
        LAM (higher resolution) reduced gaussian grid identifier, e.g. 'O320'.
    area_mask_builder : KNNAreaMaskBuilder
        KNNAreaMaskBuilder with the cloud of points to define the AOI.

    Returns
    -------
    torch.Tensor of shape (num_nodes, 2)
        The latitude and longitude coordinates, in radians.
    """
    assert area_mask_builder is not None, "AOI mask builder must be provided to build stretched grid."

    # Get the low resolution global nodes
    global_coords_rad = get_latlon_coords_gaussian(global_grid)
    LOGGER.info("Global grid %s has %d nodes.", global_grid, len(global_coords_rad))

    # Mask to keep only global nodes OUTSIDE the AOI
    global_area_mask = ~area_mask_builder.get_mask(global_coords_rad)
    global_coords_outside_aoi = global_coords_rad[global_area_mask]
    LOGGER.info("Keeping %d global nodes outside AOI.", len(global_coords_outside_aoi))

    # Get the high resolution lam nodes
    lam_coords_rad = get_latlon_coords_gaussian(lam_grid)
    LOGGER.info("LAM grid %s has %d nodes.", lam_grid, len(lam_coords_rad))

    # Mask to keep only lam nodes INSIDE the AOI
    lam_area_mask = area_mask_builder.get_mask(lam_coords_rad)
    lam_coords_inside_aoi = lam_coords_rad[lam_area_mask]
    LOGGER.info("Keeping %d LAM nodes inside AOI.", len(lam_coords_inside_aoi))

    # Concatenate: global outside AOI + lam inside AOI
    combined_coords = np.concatenate([global_coords_outside_aoi, lam_coords_inside_aoi], axis=0)
    LOGGER.info("Total nodes after combining: %d.", len(combined_coords))

    # Sort by latitude and longitude, consistent with get_coordinates_ordering
    node_ordering = get_coordinates_ordering_stable(combined_coords)

    return torch.tensor(combined_coords[node_ordering], dtype=torch.float32)

def get_coordinates_ordering_stable(coords: np.ndarray) -> np.ndarray:
    index_latitude = np.argsort(coords[:, 1])                      # sort by lon (secondary key)
    index_longitude = np.argsort(-coords[index_latitude][:, 0], kind='stable')  # sort by lat desc (primary key)
    node_ordering = np.arange(coords.shape[0])[index_latitude][index_longitude]
    return node_ordering