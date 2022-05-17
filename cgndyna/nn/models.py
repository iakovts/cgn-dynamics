import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN

class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features, 
                           out_channels=32, 
                           periods=periods)
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

#TemporalGNN(node_features=2, periods=12)

def train(signals):
    device = torch.device("cpu")
    model = TemporalGNN(node_features=1, periods=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(10):
        loss = 0
        step = 0
        for signal in signals:
            for snapshot in signal:
                snapshot = snapshot.to(device)
                y_hat = model(snapshot.x, snapshot.edge_index)
                loss = loss + torch.mean((y_hat-snapshot.y) ** 2)
                step += 1
        loss = loss / (step + 1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("Epoch {} train MSE: {:.4f}".format(epoch, loss.item()))
            
    
