from __future__ import annotations

import torch

import torch_geometric_temporal as tg

from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import dense_to_sparse
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from cgndyna.config.base import Config
    from cgndyna.util.networks.base import Network
    from cgndyna.dataset.dataset import Dataset


class DataTransformer:
    def __init__(self, config: Config, dataset: Dataset) -> None:
        self.cfg: Config = config
        self.dataset: Dataset = dataset

    def transform_dataset(self) -> tg.StaticGraphTemporalSignal:
        # t_data = from_networkx(self.dataset.nw)  # Convert to `torch_geometric.Data` obj.
        edge_indeces, values = self.get_edges_and_weights()
        trans_dataset = tg.StaticGraphTemporalSignal(
            edge_index=edge_indeces,
            edge_weight=values.float(),
            features=self.dataset.data[0, :, 0],
            targets=self.dataset.data[0, :, 1],
        )
        return trans_dataset

    def get_edges_and_weights(self) -> tuple[Any, Any]:
        adj_data_np = from_networkx(self.dataset.nw)
        adj_data = torch.from_numpy(adj_data_np)
        edge_indeces, values = dense_to_sparse(adj_data)
        edge_indeces = edge_indeces.numpy()
        values = values.numpy()
        return edge_indeces, values
