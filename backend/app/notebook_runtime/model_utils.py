"""
VFL SHAP model definitions (PyTorch modules).

Used by training and prediction notebooks; import from `utils.model_utils` or `utils` package.
"""

import torch
import torch.nn as nn


class LocalEncoder(nn.Module):
    """Local encoder for each agent in Vertical Federated Learning."""

    def __init__(self, input_dim, embed_dim=64, hidden_dim=128):
        super().__init__()
        # Deeper encoder for better feature learning
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)


class ActiveClassifier(nn.Module):
    """Active agent classifier that combines embeddings from all agents."""

    def __init__(self, embed_dim=64, num_classes=5, hidden_dim=128):
        super().__init__()
        # Deeper classifier
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, h):
        # Return logits (NO softmax) - CrossEntropyLoss applies softmax internally
        return self.net(h)


class VFLModel(nn.Module):
    """Vertical Federated Learning model with multiple agents."""

    def __init__(self, input_dims, embed_dim=64, num_classes=5, hidden_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.encoders = nn.ModuleList(
            [LocalEncoder(dim, embed_dim, hidden_dim) for dim in input_dims]
        )
        # Improved fusion: concat embeddings instead of sum
        fusion_dim = embed_dim * len(input_dims)  # 3 parties * embed_dim
        self.classifier = ActiveClassifier(fusion_dim, num_classes, hidden_dim)

    def forward(self, x_parts):
        embeddings = [enc(x) for x, enc in zip(x_parts, self.encoders)]
        # Improved fusion: concatenate instead of sum for better separation
        h = torch.cat(embeddings, dim=1)  # [B, embed_dim*3] instead of [B, embed_dim]
        y_hat = self.classifier(h)
        return y_hat

    def get_agent_embeddings(self, x_parts):
        """Get embeddings from each agent without computing final prediction."""
        self.eval()
        with torch.no_grad():
            embeddings = [enc(x) for x, enc in zip(x_parts, self.encoders)]
        return embeddings


class AgentMetaModel(nn.Module):
    """Meta-model that operates on concatenated agent embeddings."""

    def __init__(self, in_dim=192, num_classes=9, hidden_dim=128):
        super().__init__()
        # MLP instead of linear: can represent complex decision boundaries
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_meta):
        # Return logits (NO softmax) - soft distillation uses logits
        return self.net(x_meta)


class StandardNeuralNetwork(nn.Module):
    """Standard (non-federated) neural network for comparison with VFL model."""

    def __init__(self, input_dim, num_classes=9, hidden_dims=[256, 128, 64], dropout=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim

        # Build layers dynamically
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Return logits (NO softmax) - CrossEntropyLoss applies softmax internally
        return self.net(x)
