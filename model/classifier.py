"""
classifier.py

Classification Head

Converts the fused feature representation
into attack class logits.
"""

import torch
import torch.nn as nn

from .blocks import MLPBlock


class ClassificationHead(nn.Module):
    """
    Classification Head

    Input:
        Shared feature vector

    Output:
        Logits for every attack class
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dims=(32,),
        num_classes: int = 15,
        dropout: float = 0.3
    ):
        super().__init__()

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

        layers.append(
            nn.Linear(
                previous_dim,
                num_classes
            )
        )

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):

        logits = self.classifier(x)

        return logits