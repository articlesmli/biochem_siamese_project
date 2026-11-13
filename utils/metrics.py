import torch
import torch.nn.functional as F
from torch import nn


class ContrastiveLoss(nn.Module):
  """Contrastive loss function for Siamese networks.

  Based on Hadsell et al. (2006):
  L = (1 - Y) * 0.5 * D^2 + Y * 0.5 * max(0, margin - D)^2
  Where:
    - Y = 1.0 if similar, 0.0 if dissimilar
    - D = Euclidean distance between embeddings
    - margin = minimum distance enforced between dissimilar pairs
  """

  def __init__(self, margin=2.0):
    super().__init__()
    self.margin = margin

  def forward(self, output1, output2, label):
    # 1. Calculate Euclidean distance between the two embedding vectors
    euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

    # 2. Loss component for similar pairs (label = 1.0)
    loss_similar = label * torch.pow(euclidean_distance, 2)

    # 3. Loss component for dissimilar pairs (label = 0.0)
    loss_dissimilar = (1 - label) * torch.pow(
        torch.clamp(self.margin - euclidean_distance, min=0.0), 2
    )

    # 4. Combine losses and compute mean across the batch
    total_loss = 0.5 * torch.mean(loss_similar + loss_dissimilar)
    return total_loss
