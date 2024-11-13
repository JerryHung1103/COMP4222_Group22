import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
length_of_embedding = 64
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.sigmoid(self.conv1(x, edge_index))
        x = F.sigmoid(self.conv2(x, edge_index))
        x = F.sigmoid(self.conv3(x, edge_index))
        return torch.sum(x, dim=0) 