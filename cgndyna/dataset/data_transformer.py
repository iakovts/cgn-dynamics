from __future__ import annotations

import torch

import networkx as nx
import numpy as np
import torch_geometric_temporal as tg

from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import dense_to_sparse
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cgndyna.dataset.dataset import Dataset


class DatasetLoader:
    def __init__(self, dataset: list[Dataset], data_idx: int) -> None:
        self.nw: nx.Graph = dataset[data_idx].nw
        self.data: np.ndarray = dataset[
            data_idx
        ].data  # array shaped (n_sample, n_nodes, lag)

    def _get_edges_and_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns a tuple containing numpy arrays of the `edge_indeces` and
        `edge_weights` in a format compatible with StaticGraphTemporalSignal.
        """
        G = self.nw
        adj_np = nx.to_numpy_matrix(G)
        edge_indeces, edge_weights = dense_to_sparse(torch.from_numpy(adj_np))
        edge_indeces = edge_indeces.numpy()
        edge_weights = edge_weights.numpy()
        return edge_indeces, edge_weights

    def get_dataset(self, lags: int = 2) -> list[tg.StaticGraphTemporalSignal]:
        samples = self.data.shape[0]
        signals = []
        for sample in range(samples):
            edge_indeces, edge_weights = self._get_edges_and_weights()
            # Expand dims here to account for node features.
            features = [
                np.expand_dims(self.data[sample, :, i : i + lags], axis=1)
                for i in range(self.data.shape[2] - lags)
            ]
            targets = [
                np.expand_dims(self.data[sample, :, i + lags], axis=1)
                for i in range(self.data.shape[2] - lags)
            ]
            signals.append(
                tg.StaticGraphTemporalSignal(
                    edge_index=edge_indeces,
                    edge_weight=edge_weights,
                    features=features,
                    targets=targets,
                )
            )
        return signals
