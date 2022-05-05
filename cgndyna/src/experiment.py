from cgndyna.config.base import Config
from cgndyna.config.base import ExperimentCfg, NetworkCfg
from cgndyna.util.networks.common import SimpleNetwork


class Experiment:
    def __init__(self, config):
        self.cfg = config
        self.nw_gen = None
        self.networks = None

    def setup(self):
        """Initializes experiment's parameters"""
        self.set_networks()

    def set_networks(self):
        """Generates networks based on config, and populates the
        instance's `networks` attribute
        """
        net_gen = SimpleNetwork(self.cfg).network_generator()
        self.networks = [next(net_gen) for nw in range(self.cfg.exp.num_networks)]


def test_only():
    c = Config()
    c.nw = NetworkCfg("gnp", 0.004)
    c.exp = ExperimentCfg(num_networks=10)
    return Experiment(c)
