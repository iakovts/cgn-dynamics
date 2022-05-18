from __future__ import annotations

import numpy as np
from typing import Generator, TYPE_CHECKING, Any, Optional, NewType

from cgndyna.util.networks.common import SimpleNetwork
from cgndyna.util.dynamics.majority import MajorityRule
from cgndyna.config.base import Config, ExperimentCfg, NetworkCfg, DynamicsCfg
from cgndyna.dataset.dataset import Dataset
from cgndyna.dataset.data_transformer import DatasetLoader
from cgndyna.nn.models import TestModel

if TYPE_CHECKING:
    import torch_geometric_temporal as tg
    Signals = Optional[dict[int, list[tg.StaticGraphTemporalSignal]]]


class Experiment:
    def __init__(self, config: Config) -> None:
        self.cfg: Config = config
        self.nw_gen: Generator
        self.networks: list[SimpleNetwork]
        self.dynamics: MajorityRule
        self.dataset: list[Dataset] = []
        self.signals: Signals

    def setup(self) -> None:
        """Initializes experiment's parameters"""
        self.set_networks()

    def set_networks(self) -> None:
        """Generates networks based on config, and populates the
        instance's `networks` attribute.
        """
        net_gen = SimpleNetwork(self.cfg).network_generator()
        self.networks = [*net_gen]

    def set_dynamics(self, nw: SimpleNetwork) -> None:
        """Sets the dynamics attribute for the experiment."""
        self.dynamics = MajorityRule(self.cfg, nw)

    def generate_data(self) -> None:
        """Populates the `self.dataset` attr with `Dataset` objects
        containing the dynamics ran on each network. Datasets have shape
        (samples, nodes, lagsteps) and also contain network info.
        """
        self.setup()
        for nw in range(self.cfg.exp.num_networks):
            dataset = Dataset(self.networks[nw], self.cfg)
            self.set_dynamics(self.networks[nw])
            for sample in range(self.cfg.exp.num_samples):
                x = self.dynamics.initial_state()
                # import pdb; pdb.set_trace()
                dataset.data[sample, :, 0] = x
                for lagstep in range(1, self.cfg.exp.lag):
                    if lagstep % self.cfg.exp.lagstep == 0:
                        dataset.data[sample, :, lagstep] = x = self.dynamics.step(x)
            self.dataset.append(dataset)

    def transform_data(self) -> None:
        self.signals = dict()
        for nw_idx in range(self.cfg.exp.num_networks):
            self.signals[nw_idx] = DatasetLoader(self.dataset, nw_idx).get_dataset()


def test_only():
    c = Config()
    c.nw = NetworkCfg("gnp", 0.004)
    c.exp = ExperimentCfg(lag=5, num_networks=5, num_samples=100)
    c.dyn = DynamicsCfg("major", [0.4, 0.6])
    e = Experiment(c)
    e.generate_data()
    e.transform_data()
    tm = TestModel(e.signals)
    tm.setup()
    tm.train()
    labels, preds = tm.evaluate()
    return e, labels, preds


e = test_only()
# train(e.signals)
