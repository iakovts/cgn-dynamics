import numpy as np

from cgndyna.util.dynamics.base import Dynamics


class MajorityRule(Dynamics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_nodes = self.cfg.nw.num_nodes
        self.init_param = self.cfg.dyn.init_param

    def initial_state(self):
        self.get_neighbors_state()
        rng = np.random.default_rng()
        x = rng.choice([0, 1], p=self.init_param, size=self.num_nodes)
        return x

    def step(self, x):
        y = np.copy(x)
        for node in range(self.num_nodes):
            import pdb; pdb.set_trace()
            major = np.sum(x[self.neighbors[node]])
            if major > (len(self.neighbors[node]) / 2):
                y[node] = 1
            elif major < (len(self.neighbors[node]) / 2):
                y[node] = 0
        # f = np.zeros((self.num_nodes, 2))
        # f[y == 0, 0] = 1
        # f[y == 1, 1] = 1
        return y
