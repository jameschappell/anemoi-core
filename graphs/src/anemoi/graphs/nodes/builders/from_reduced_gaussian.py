# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import re
import os
import time
from abc import abstractmethod

import networkx as nx
import torch
from torch_geometric.data import HeteroData
from abc import ABC
import numpy as np

from anemoi.graphs.nodes.builders.base import BaseNodeBuilder
from anemoi.graphs.generate.masks import KNNAreaMaskBuilder

from anemoi.utils.grids import grids


LOGGER = logging.getLogger(__name__)


class ReducedGaussianGridNodes(BaseNodeBuilder, ABC):
    """Nodes from a reduced gaussian grid.

    A gaussian grid is a latitude/longitude grid. The spacing of the latitudes is not regular. However, the spacing of
    the lines of latitude is symmetrical about the Equator. A grid is usually referred to by its 'number' N/O, which
    is the number of lines of latitude between a Pole and the Equator. The N code refers to the original ECMWF reduced
    Gaussian grid, whereas the code O refers to the octahedral ECMWF reduced Gaussian grid.

    Attributes
    ----------
    grid : str
        The reduced gaussian grid, of shape {n,N,o,O}XXX with XXX latitude lines between the pole and
        equator.

    Methods
    -------
    get_coordinates()
        Get the lat-lon coordinates of the nodes.
    register_nodes(graph, name)
        Register the nodes in the graph.
    register_attributes(graph, name, config)
        Register the attributes in the nodes of the graph specified.
    update_graph(graph, name, attrs_config)
        Update the graph with new nodes and attributes.
    """

    def __init__(self, grid: int, name: str) -> None:
        """Initialize the ReducedGaussianGridNodes builder."""
        assert re.fullmatch(
            r"^[oOnN]\d+$", grid
        ), f"{self.__class__.__name__}.grid must match the format [n|N|o|O]XXX with XXX latitude lines between the pole and equator."
        self.grid = grid
        super().__init__(name)

    def get_coordinates(self) -> torch.Tensor:
        """Get the coordinates of the nodes.

        Returns
        -------
        torch.Tensor of shape (num_nodes, 2)
            A 2D tensor with the coordinates, in radians.
        """
        return self.create_nodes()
    
    @abstractmethod
    def create_nodes(self) -> torch.Tensor: ...
 
    
class ReducedGaussianNodes(ReducedGaussianGridNodes):
    
    def create_nodes(self) -> torch.Tensor:
        # Synchronize downloads across distributed ranks
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            
            # Rank 0 downloads first and caches the data
            if rank == 0:
                LOGGER.info(f"Rank 0: Downloading grid data for {self.grid}")
                grid_data = grids(self.grid)
            
            # Barrier to ensure rank 0 completes download before other ranks proceed
            torch.distributed.barrier()
            
            # Other ranks can now access cached data
            if rank != 0:
                grid_data = grids(self.grid)
        else:
            # Non-distributed case
            # Check if we're in a multi-process environment (even if distributed not yet initialized)
            local_rank = os.environ.get("LOCAL_RANK")
            if local_rank is not None and int(local_rank) != 0:
                # Wait for rank 0 to download and cache the data
                LOGGER.info(f"Rank {local_rank}: Waiting for rank 0 to cache grid data for {self.grid}")
                time.sleep(3)
            grid_data = grids(self.grid)
        
        coords = self.reshape_coords(grid_data["latitudes"], grid_data["longitudes"])
        return coords
       
    
class LimitedAreaReducedGaussianGridNodes(ReducedGaussianGridNodes, ABC):
    """Nodes based on reduced gaussian grids using an area of interest.

    Attributes
    ----------
    area_mask_builder : KNNAreaMaskBuilder
        The area of interest mask builder.
    """

    def __init__(
        self,
        grid: str,
        reference_node_name: str,
        name: str,
        mask_attr_name: str | None = None,
        margin_radius_km: float = 100.0,
    ) -> None:

        super().__init__(grid, name)
        self.hidden_attributes = self.hidden_attributes | {"area_mask_builder"}

        self.area_mask_builder = KNNAreaMaskBuilder(reference_node_name, margin_radius_km, mask_attr_name)

    def register_nodes(self, graph: HeteroData) -> None:
        self.area_mask_builder.fit(graph)
        return super().register_nodes(graph)


class StretchedReducedGaussianGridNodes(LimitedAreaReducedGaussianGridNodes, ABC):
    """Nodes based on reduced gaussian grids using an area of interest, with a higher resolution region
    in the area of interest.

    Attributes
    ----------
    area_mask_builder : KNNAreaMaskBuilder
        The area of interest mask builder.
    """

    def __init__(
        self,
        global_grid: int,
        lam_grid: int,
        name: str,
        reference_node_name: str,
        mask_attr_name: str | None = None,
        margin_radius_km: float = 100.0,
    ) -> None:
        super().__init__(
            grid=lam_grid,
            reference_node_name=reference_node_name,
            mask_attr_name=mask_attr_name,
            margin_radius_km=margin_radius_km,
            name=name,
        )
        self.global_grid = global_grid
        

class StretchedReducedGaussianNodes(StretchedReducedGaussianGridNodes):
    """
    Nodes from two reduced gaussian grids - a coarser global grid with a
    higher resolution region in the area of interest.
    """

    def create_nodes(self) -> torch.Tensor:
        from anemoi.graphs.generate.reduced_gaussian import create_stretched_reduced_gaussian_nodes

        return create_stretched_reduced_gaussian_nodes(
            global_grid=self.global_grid,
            lam_grid=self.grid,
            area_mask_builder=self.area_mask_builder,
        )