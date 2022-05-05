import networkx as nx

from cgndyna.util.networks.base import Network


network_funcs = {
    "gnp": nx.fast_gnp_random_graph,
    "ba": nx.barabasi_albert_graph,
}


class SimpleNetwork(Network):
    def __init__(self, config):
        super().__init__(config)

    def network_generator(self):
        """Returns a generator object creating graphs, based on given
        configuration. Pulls the name of the needed graph from config
        as well as the rest of the parameters and chooses the right nx
        function.
        """
        while True:
            graph = network_funcs[self.cfg.nw.name](
                self.cfg.nw.num_nodes, self.cfg.nw.nw_param, self.cfg.nw.seed
            )
            yield graph
