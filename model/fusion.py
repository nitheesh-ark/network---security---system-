"""
fusion.py

Feature Fusion Network

This module combines the embeddings produced by all
feature branches into a single shared representation.
"""

import torch
import torch.nn as nn

from .blocks import MLPBlock


class FeatureFusion(nn.Module):
    """
    Feature Fusion Module

    Input:
        Flow Embedding
        Packet Embedding
        Time Embedding
        Flags Embedding
        Header Embedding

    Process:
        Concatenate
            ↓
        MLPBlock
            ↓
        MLPBlock
            ↓
        Shared Feature Vector

    Output:
        (batch_size, fusion_dim)
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        num_branches: int = 5,
        hidden_dims=[128, 64],
        dropout: float = 0.3
    ):
        super().__init__()

        input_dim = embedding_dim * num_branches

        layers = []

        previous_dim = input_dim

        for hidden_dim in hidden_dims:

            layers.append(
                MLPBlock(
                    previous_dim,
                    hidden_dim,
                    dropout
                )
            )

            previous_dim = hidden_dim

        self.fusion = nn.Sequential(*layers)

        self.output_dim = previous_dim

    def forward(
        self,
        flow,
        packet,
        time,
        flags,
        header
    ):

        # Concatenate along feature dimension
        x = torch.cat(
            [
                flow,
                packet,
                time,
                flags,
                header
            ],
            dim=1
        )

        x = self.fusion(x)

        return x