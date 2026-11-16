import torch
import torch.nn.functional as F
from torch import nn


class ContrastiveLoss(nn.Module):
  """Contrastive loss function for Siamese networks.

  Matches SiameseBiochemDataset label convention:
    - label = 0.0 if similar (same BBBP class)
    - label = 1.0 if dissimilar (different BBBP class)
    - D = Euclidean distance between L2-normalized embeddings (range [0, 2])
    - margin = minimum distance enforced between dissimilar pairs
  """

  def __init__(self, margin=1.0):
    super().__init__()
    self.margin = margin

  def forward(self, output1, output2, label):
    # Ensure 1D shape (B,) to match label shape (B,)
    euclidean_distance = F.pairwise_distance(output1, output2, keepdim=False)

    # Loss component for similar pairs (label = 0.0): pull embeddings together
    loss_similar = (1.0 - label) * torch.pow(euclidean_distance, 2)

    # Loss component for dissimilar pairs (label = 1.0): push embeddings past margin
    loss_dissimilar = label * torch.pow(
        torch.clamp(self.margin - euclidean_distance, min=0.0), 2
    )

    total_loss = 0.5 * torch.mean(loss_similar + loss_dissimilar)
    return total_loss