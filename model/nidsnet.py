"""
nidsnet.py

Complete Network Intrusion Detection System (NIDSNet)

Architecture

 Flow Features (18)
 Packet Features (12)
 Time Features (6)
 Flags Features (9)
 Header Features (14)

            │
            ▼

      Feature Branches

            │
            ▼

      Feature Fusion

            │
            ▼

   Classification Head

            │
            ▼

      15 Attack Classes
"""

import torch
import torch.nn as nn

from .branches import FeatureBranch
from .fusion import FeatureFusion
from .classifier import ClassificationHead


class NIDSNet(nn.Module):

    def __init__(
        self,
        embedding_dim=16,
        hidden_dims=[64, 32],
        fusion_hidden=[128, 64],
        num_classes=15,
        dropout=0.3
    ):

        super().__init__()

        ##################################################
        # Feature Branches
        ##################################################
        self.flow_branch = FeatureBranch(
            input_dim=10,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            dropout=dropout
        )

        self.packet_branch = FeatureBranch(
            input_dim=18,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            dropout=dropout
        )

        self.time_branch = FeatureBranch(
            input_dim=22,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            dropout=dropout
        )

        self.flags_branch = FeatureBranch(
            input_dim=12,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            dropout=dropout
        )

        self.header_branch = FeatureBranch(
            input_dim=15,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            dropout=dropout
        )

        ##################################################
        # Feature Fusion
        ##################################################

        self.feature_fusion = FeatureFusion(
            embedding_dim=embedding_dim,
            num_branches=5,
            hidden_dims=fusion_hidden,
            dropout=dropout
        )

        ##################################################
        # Classification Head
        ##################################################

        self.classifier = ClassificationHead(
            input_dim=self.feature_fusion.output_dim,
            hidden_dims=(32,),
            num_classes=num_classes,
            dropout=dropout
        )

    ##################################################
    # Forward
    ##################################################

    def forward(
        self,
        flow,
        packet,
        time,
        flags,
        header
    ):

        ##################################################
        # Branch Embeddings
        ##################################################

        flow_embedding = self.flow_branch(flow)

        packet_embedding = self.packet_branch(packet)

        time_embedding = self.time_branch(time)

        flags_embedding = self.flags_branch(flags)

        header_embedding = self.header_branch(header)

        ##################################################
        # Fusion
        ##################################################

        fused_features = self.feature_fusion(
            flow_embedding,
            packet_embedding,
            time_embedding,
            flags_embedding,
            header_embedding
        )

        ##################################################
        # Classification
        ##################################################

        logits = self.classifier(fused_features)

        return logits