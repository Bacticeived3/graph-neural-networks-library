import torch
from torch_geometric.nn import GCNConv
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)