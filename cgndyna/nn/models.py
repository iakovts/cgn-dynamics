from __future__ import annotations


from itertools import chain
from random import shuffle

import torch
import torch.nn.functional as F
import numpy as np

from torch_geometric_temporal.nn.recurrent import A3TGCN

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch_geometric_temporal as tg

    Signals_D = Optional[dict[int, list[tg.StaticGraphTemporalSignal]]]
    Signal = list[tg.StaticGraphTemporalSignal]
    Model = torch.nn.Module


class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features, out_channels=32, periods=periods)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h


class TestModel:
    def __init__(self, signals: Signals_D, train_ratio: float = 0.8):
        self.signals = signals
        self.train_ratio = train_ratio
        self.train_set: Signal
        self.eval_set: Signal
        self.model: Model
        self.device: str = "cpu"

    def setup(self):
        self.split_data()
        self.model = TemporalGNN(node_features=1, periods=1).to(self.device)

    def split_data(self):
        merged = list(chain(*self.signals.values()))
        shuffle(merged)
        self.train_set = merged[: int(len(merged) * self.train_ratio)]
        self.eval_set = merged[int(len(merged) * self.train_ratio) :]

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.model.train()
        for epoch in range(10):
            loss = 0
            step = 0
            for signal in self.train_set:
                for snapshot in signal:
                    snapshot = snapshot.to(self.device)
                    y_hat = self.model(snapshot.x, snapshot.edge_index)
                    loss = loss + torch.mean((y_hat - snapshot.y) ** 2)
                    step += 1
            loss = loss / (step + 1)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("Epoch {} train MSE: {:.4f}".format(epoch, loss.item()))

    def evaluate(self):
        self.model.eval()
        loss = 0
        step = 0
        predictions = []
        labels = []
        for signal in self.eval_set:
            for snapshot in signal:
                snapshot = snapshot.to(self.device)
                y_hat = self.model(snapshot.x, snapshot.edge_index)
                loss = loss + torch.mean((y_hat - snapshot.y) ** 2)
                labels.append(snapshot.y)
                predictions.append(y_hat)
                step += 1
        loss = loss / (step + 1)
        loss = loss.item()
        print("Test MSE: {:.4f}".format(loss))
        return labels, predictions
