import networkx as nx

from dataclasses import dataclass, field, make_dataclass

from typing import Union, List




@dataclass
class NetworkCfg:
    name: str
    nw_param: Union[int, float]  # Network parameter i.e.: `p` for GNP or `m` for BA
    num_nodes: int = field(default=1000)
    seed: Union[int, float] = field(default=None)
    # is_modular: bool = field(default=False)

    # def __post_init__(self):
    #     if self.is_modular:

@dataclass
class DynamicsCfg:
    name: str
    init_param: List[float]

@dataclass
class ExperimentCfg:
    num_samples: int = field(default=100)
    num_networks: int = field(default=1)
    lag: int = field(default=1)
    lagstep: int = field(default=1)


@dataclass
class Config:
    nw: NetworkCfg = field(init=False)
    exp: ExperimentCfg = field(init=False)
    dyn: DynamicsCfg = field(init=False)
