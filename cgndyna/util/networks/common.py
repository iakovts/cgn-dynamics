from __future__ import annotations

import networkx as nx

from cgndyna.util.networks.base import Network

from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from cgndyna.config.base import Config
    from networkx.classes.graph import Graph

network_funcs = {
    "gnp": nx.fast_gnp_random_graph,
    "ba": nx.barabasi_albert_graph,
}


class SimpleNetwork(Network):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def network_generator(self) -> Generator[Graph, None, None]:
        """Returns a generator object creating graphs, based on given
        configuration. Pulls the name of the needed graph from config
        as well as the rest of the parameters and chooses the right nx
        function.
        """
        for nw in range(self.cfg.exp.num_networks):
            graph = network_funcs[self.cfg.nw.name](
                self.cfg.nw.num_nodes, self.cfg.nw.nw_param, self.cfg.nw.seed
            )
            yield graph
