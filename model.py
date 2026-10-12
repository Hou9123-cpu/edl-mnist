"""A small convolutional network with an evidential (Dirichlet) output head.

Following Sensoy et al. (2018), the network emits non-negative *evidence* e_c
for each class c. The parameters of a Dirichlet distribution over class
probabilities are then alpha_c = e_c + 1, and the expected class probabilities
are p_c = alpha_c / sum_c alpha_c.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import N_CLASSES


class EvidentialCNN(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                      # 16 x 14 x 14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                      # 32 x 7 x 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        logits = self.classifier(self.features(x))
        evidence = F.softplus(logits)            # non-negative evidence
        alpha = evidence + 1.0                    # Dirichlet concentration parameters
        return alpha, logits


def predict_proba(alpha):
    """Expected class probabilities under the Dirichlet: p_c = alpha_c / S."""
    S = alpha.sum(dim=-1, keepdim=True)
    return alpha / S
