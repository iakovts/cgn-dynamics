import networkx as nx

from dataclasses import dataclass, field, make_dataclass

from typing import Union




@dataclass
class Network:
    name: str
    nw_param: Union[int, float]  # Network parameter i.e.: `p` for GNP or `m` for BA
    num_nodes: int = field(default=1000)
    # is_modular: bool = field(default=False)

    # def __post_init__(self):
    #     if self.is_modular:

@dataclass
class Config:
    nw: Network = field(init=False)
