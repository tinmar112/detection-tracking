import torch
from torch import nn


class PFNLayer(nn.Module):

    def __init__(self, in_channels=9, out_channels=64):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [P, Np_max, 9]

        x = self.linear(x)

        # BatchNorm expects [batch, channels, ...]
        x = x.transpose(1, 2) # x: [P, 9, Np_max] (normalising along x, y, z, etc.)
        x = self.norm(x)
        x = x.transpose(1, 2) # back to x: [P, Np_max, 9]

        x = self.relu(x)

        # aggregating points within each pillar -> keep max of pillar
        x = torch.max(x, dim=1)[0]

        # x: [P, 64]
        return x
