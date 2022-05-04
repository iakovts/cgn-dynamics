from abc import ABC, abstractmethod

import numpy as np


class Dynamics(ABC):
    def __init__(self, config=None):
        self.config = config

    @abstractmethod
    def initial_state(self):
        pass
    
