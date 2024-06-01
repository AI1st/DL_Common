import torch
from torch import nn


class LayerNormAuto(nn.Module):
    def __init__(self, ignore_dim):
        super().__init__()
        self.ln = None
        self.ignore_dim = ignore_dim

    def forward(self, X):
        if self.ln is None:
            self.init_ln(X)
        return self.ln(X)

    def init_ln(self, X):
        self.ln = nn.LayerNorm(X.shape[self.ignore_dim:]).to(X.device)
