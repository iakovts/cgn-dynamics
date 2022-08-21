from __future__ import annotations

import numpy as np

from cgndyna.util.dynamics.base import Dynamics

class SIS(Dynamics):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_nodes: int = self.cfg.nw.num_nodes
        self.init_param : list[float] = self.cfg.dyn.init_param
        self.infection = 0.08
        self.recovery = 0.08
        self.get_neighbors()


    def initial_state(self) -> np.ndarray:
        rng = np.random.default_rng()
        x = rng.choice([0, 1], p=self.init_param, size=self.num_nodes)
        return x

    def infect_prob(self, inf_neigh):
        return 1 - ((1 - self.infection) ** inf_neigh)
    
    def step(self, x) -> np.ndarray:
        y = np.copy(x)
        for node in range(self.num_nodes):
            if y[node] == 0:
                inf_chance = self.infect_prob(np.sum(x[self.neighbors[node]]))
                y[node] = np.random.choice([0, 1], p=[1 - inf_chance, inf_chance])
            else:
                y[node] = np.random.choice([0, 1], p=[self.recovery, 1-self.recovery])
        return y

