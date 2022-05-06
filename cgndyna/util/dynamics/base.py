from abc import ABC, abstractmethod

import numpy as np


class Dynamics(ABC):
    def __init__(self, config, network):
        self.cfg = config
        self.nw = network

    @abstractmethod
    def initial_state(self):
        """Defines the initial state of the dynamics on the network."""
        ...

    @abstractmethod
    def step(self, x):
        """Progress to the next "step" of the dynamics."""

    def get_neighbors_state(self):
        """Sets the state (feature vector) of the neighbors of a node
        in a dictionary."""
        self.neighbors = {
            node: np.array(self.nw.neighbors(node)) for node in range(len(self.nw))
        }
