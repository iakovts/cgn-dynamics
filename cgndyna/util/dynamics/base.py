from abc import ABC, abstractmethod

import numpy as np


class Dynamics(ABC):
    def __init__(self, config=None):
        self.cfg = config

    @abstractmethod
    def initial_state(self):
        pass

    def get_neighbors_state(self):
        self.neighbors = {}
    
