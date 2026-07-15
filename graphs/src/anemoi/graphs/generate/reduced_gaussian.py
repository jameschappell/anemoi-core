# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import re

import numpy as np
import torch
from requests.exceptions import HTTPError

from anemoi.graphs.generate.masks import AreaMaskBuilder
from anemoi.utils.grids import grids

LOGGER = logging.getLogger(__name__)

try:
    import eccodes
except Exception as exc:
    LOGGER.warning(
        "ecCodes Python module is unavailable or failed to initialize (%s). " "Continuing without ecCodes fallback.",
        exc,
    )
    eccodes = None


def _parse_gaussian_grid(grid: str) -> tuple[str, int]:
    match = re.match(r"^([NnOo])(\d+)$", grid)
    if not match:
        raise ValueError(f"Grid '{grid}' does not match expected format 'N{{n_points}}' or 'O{{n_points}}'.")
    return match.group(1).upper(), int(match.group(2))


def _as_numpy_1d(values, dtype):
    """Robustly convert ecCodes return values (including cdata arrays) to 1-D NumPy arrays."""
    if isinstance(values, np.ndarray):
        return values.astype(dtype, copy=False).ravel()

    try:
        return np.asarray(values, dtype=dtype).ravel()
    except Exception:
        pass

    try:
        return np.asarray(list(values), dtype=dtype).ravel()
    except Exception:
        pass

    try:
        n = len(values)
    except Exception as exc:
        raise TypeError(f"Could not determine length of values of type {type(values)}") from exc

    try:
        return np.fromiter((values[i] for i in range(n)), dtype=dtype, count=n)
    except Exception as exc:
        raise TypeError(f"Could not convert values of type {type(values)} to numpy array") from exc


def _latlon_from_pl(latitudes_deg, pl, dtype=np.float64):
    latitudes_deg = _as_numpy_1d(latitudes_deg, dtype=dtype)
    pl = _as_numpy_1d(pl, dtype=np.int64)

    if latitudes_deg.size != pl.size:
        raise ValueError(f"PL length mismatch: len(latitudes)={latitudes_deg.size}, len(pl)={pl.size}.")

    lats = np.repeat(latitudes_deg, pl)

    n_total = int(pl.sum())
    starts = np.cumsum(np.r_[0, pl[:-1]])
    idx_in_ring = np.arange(n_total) - np.repeat(starts, pl)
    nlon_per_point = np.repeat(pl, pl)

    lons = (idx_in_ring * (360.0 / nlon_per_point)).astype(dtype, copy=False)
    lons = np.where(lons > 180.0, lons - 360.0, lons)
    return lats, lons


def _eccodes_latlon_coords(grid: str, dtype=np.float64):
    if eccodes is None:
        return None

    try:
        grid_type, n_points = _parse_gaussian_grid(grid)
    except ValueError:
        return None

    try:
        latitudes_deg = _as_numpy_1d(eccodes.codes_get_gaussian_latitudes(n_points), dtype=dtype)
    except Exception as exc:
        LOGGER.warning(
            "ecCodes failed to get Gaussian latitudes for grid '%s': %s",
            grid,
            exc,
        )
        return None

    expected_latitudes = 2 * n_points
    if latitudes_deg.size != expected_latitudes:
        LOGGER.warning(
            "ecCodes returned %d latitudes for grid '%s'; expected %d.",
            latitudes_deg.size,
            grid,
            expected_latitudes,
        )
        return None

    if grid_type == "N":
        sample_name = f"reduced_gg_pl_{n_points}_grib2"
        gid = None
        try:
            gid = eccodes.codes_grib_new_from_samples(sample_name)
            pl = _as_numpy_1d(eccodes.codes_get_array(gid, "pl"), dtype=np.int64)
        except Exception as exc:
            LOGGER.warning(
                "ecCodes failed to read sample '%s' for grid '%s': %s",
                sample_name,
                grid,
                exc,
            )
            return None
        finally:
            if gid is not None:
                eccodes.codes_release(gid)
    else:
        nlons_half = 20 + 4 * np.arange(0, n_points, dtype=np.int64)
        pl = np.concatenate([nlons_half, nlons_half[::-1]])

    try:
        return _latlon_from_pl(latitudes_deg, pl, dtype=dtype)
    except Exception as exc:
        LOGGER.warning(
            "Failed to build coordinates from ecCodes data for grid '%s': %s",
            grid,
            exc,
        )
        return None


def _stack_coords_radians(lats_deg, lons_deg) -> np.ndarray:
    return np.stack([np.deg2rad(lats_deg), np.deg2rad(lons_deg)], axis=-1)


def _local_octahedral_latlon_coords(grid: str):
    try:
        grid_type, n_points = _parse_gaussian_grid(grid)
    except ValueError:
        return None

    if grid_type != "O":
        return None

    try:
        LOGGER.warning(
            "Falling back to local octahedral_reduced_gaussian_gridpoints(n_points=%d) for grid '%s'.",
            n_points,
            grid,
        )
        return octahedral_reduced_gaussian_gridpoints(n_points=n_points)
    except Exception as exc:
        LOGGER.warning(
            "Local octahedral fallback failed for grid '%s': %s",
            grid,
            exc,
        )
        return None


def get_latlon_coords_gaussian(grid: str) -> np.ndarray:
    """Get the latitude and longitude coordinates (in radians) of a reduced gaussian grid.

    Parameters
    ----------
    grid : str
        The reduced gaussian grid identifier, e.g. 'O96', 'N320'.
        Resolution order: registry lookup -> local octahedral construction
        for O-grids -> ecCodes fallback for N/O grids.

    Returns
    -------
    np.ndarray of shape (num_nodes, 2)
        The latitude and longitude coordinates, in radians.
    """
    # 1) Primary registry lookup
    try:
        grid_data = grids(grid)
        return _stack_coords_radians(grid_data["latitudes"], grid_data["longitudes"])
    except HTTPError as exc:
        LOGGER.warning(
            "Grid '%s' not found in registry (%s). Trying fallback methods.",
            grid,
            exc,
        )

    # 2) Existing local octahedral fallback for O-grids
    local_coords = _local_octahedral_latlon_coords(grid)
    if local_coords is not None:
        return _stack_coords_radians(*local_coords)

    # 3) ecCodes fallback for N/O grids
    eccodes_coords = _eccodes_latlon_coords(grid)
    if eccodes_coords is not None:
        LOGGER.warning(
            "Using ecCodes fallback coordinates for grid '%s'.",
            grid,
        )
        return _stack_coords_radians(*eccodes_coords)

    raise ValueError(f"Grid '{grid}' could not be resolved from registry, local fallback, or ecCodes fallback.")


def octahedral_reduced_gaussian_gridpoints(n_points=96, dtype=np.float64):
    """Generate coordinates for the ECMWF octahedral reduced Gaussian grid."""
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
    starts = np.cumsum(np.r_[0, nlons[:-1]])  # start index per latitude ring
    idx_in_ring = np.arange(n_total) - np.repeat(starts, nlons)
    nlon_per_point = np.repeat(nlons, nlons)

    lons = (idx_in_ring * (360.0 / nlon_per_point)).astype(dtype, copy=False)
    lons = np.where(lons > 180.0, lons - 360.0, lons)  # convert to [-180, 180]
    return lats, lons


def create_stretched_reduced_gaussian_nodes(
    global_grid: str,
    lam_grid: str,
    area_mask_builder: AreaMaskBuilder,
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
    area_mask_builder : AreaMaskBuilder
        AreaMaskBuilder with the cloud of points to define the AOI.

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
    index_latitude = np.argsort(coords[:, 1])  # sort by lon (secondary key)
    index_longitude = np.argsort(-coords[index_latitude][:, 0], kind="stable")  # sort by lat desc (primary key)
    node_ordering = np.arange(coords.shape[0])[index_latitude][index_longitude]
    return node_ordering
