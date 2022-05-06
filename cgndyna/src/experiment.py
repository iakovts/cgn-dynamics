import numpy as np

from cgndyna.config.base import Config, ExperimentCfg, NetworkCfg, DynamicsCfg
from cgndyna.util.networks.common import SimpleNetwork
from cgndyna.util.dynamics.majority import MajorityRule


class Experiment:
    def __init__(self, config):
        self.cfg = config
        self.nw_gen = None
        self.networks = None
        self.dynamics = None
        self.data = []

    def setup(self):
        """Initializes experiment's parameters"""
        self.set_networks()

    def set_networks(self):
        """Generates networks based on config, and populates the
        instance's `networks` attribute
        """
        net_gen = SimpleNetwork(self.cfg).network_generator()
        self.networks = [next(net_gen) for nw in range(self.cfg.exp.num_networks)]

    def set_dynamics(self, nw):
        """Sets the dynamics attribute for the experiment."""
        self.dynamics = MajorityRule(self.cfg, nw)

    def generate_data(self):
        self.setup()
        n_samples = self.cfg.exp.num_samples
        for nw in range(self.cfg.exp.num_networks):
            dataset = np.zeros((n_samples, self.cfg.nw.num_nodes, self.cfg.exp.lag))
            self.set_dynamics(self.networks[nw])
            for sample in range(self.cfg.exp.num_samples):
                x = self.dynamics.initial_state()
                dataset[sample, :, 0] = x
                for lagstep in range(1, self.cfg.exp.lag):
                    if lagstep % self.cfg.exp.lagstep == 0:
                        dataset[sample, :, lagstep] = self.dynamics.step(x)
            self.data.append(dataset)


def test_only():
    c = Config()
    c.nw = NetworkCfg("gnp", 0.004)
    c.exp = ExperimentCfg(lag=5,num_networks=10)
    c.dyn = DynamicsCfg("major", [0.4, 0.6])
    return Experiment(c)
