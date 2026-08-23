"""
branches.py

Contains the FeatureBranch module.

Each branch learns a latent representation
for one feature group.

Flow Features
Packet Features
Time Features
Flags Features
Header Features

All branches share the same architecture but
have different input dimensions.
"""

import torch
import torch.nn as nn

from .blocks import MLPBlock


class FeatureBranch(nn.Module):
    """
    Generic Feature Encoder.

    Example
    -------

    Input
        ↓
    MLPBlock
        ↓
    MLPBlock
        ↓
    MLPBlock
        ↓
    Embedding

    Output Shape
    ------------
    (batch_size, embedding_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        embedding_dim: int,
        dropout: float = 0.2
    ):
        super().__init__()

        layers = []

        previous_dim = input_dim

        # Hidden Layers
        for hidden_dim in hidden_dims:

            layers.append(
                MLPBlock(
                    previous_dim,
                    hidden_dim,
                    dropout
                )
            )

            previous_dim = hidden_dim

        # Final Projection Layer
        layers.append(
            nn.Linear(
                previous_dim,
                embedding_dim
            )
        )

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):

        embedding = self.encoder(x)

        return embedding