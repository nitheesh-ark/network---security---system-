import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """
    Basic building block used throughout the network.

    Architecture:
        Linear
          ↓
      BatchNorm1d
          ↓
         GELU
          ↓
       Dropout
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.2
    ):
        super().__init__()

        self.block = nn.Sequential(

            nn.Linear(
                in_features,
                out_features
            ),

            nn.BatchNorm1d(
                out_features
            ),

            nn.GELU(),

            nn.Dropout(
                p=dropout
            )

        )

    def forward(self, x):

        return self.block(x)