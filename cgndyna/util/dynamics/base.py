from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

import numpy.typing as npt

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cgndyna.config.base import Config
    from networkx import Graph


class Dynamics(ABC):
    def __init__(self, config: Config, network: Graph) -> None:
        self.cfg = config
        self.nw = network
        self.neighbors: dict[int, np.ndarray]

    @abstractmethod
    def initial_state(self) -> np.ndarray:
        """Defines the initial state of the dynamics on the network."""
        ...

    @abstractmethod
    def step(self, x: np.ndarray) -> np.ndarray:
        """Progress to the next "step" of the dynamics."""

    def get_neighbors(self) -> None:
        """Sets the value of self.neighbors to a dict of
        node: np.array(node's neighbors) pairs. If a node doesn't have
        any neighbors, it is assigned as a neighbor to itself instead
        """
        self.neighbors = {
            node: np.array(self.nw[node]) if self.nw[node] else np.array([node])
            for node in range(len(self.nw))
        }
