import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):

  def __init__(self):
    super(SiameseNetwork, self).__init__()

    # Convolutional Feature Extraction Subnetwork
    self.cnn = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),  # 128x128 -> 64x64

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),  # 64x64 -> 32x32

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
    )

    # Fully Connected Projection Head
    self.fc = nn.Sequential(
        nn.Linear(128 * 16 * 16, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 128),  # Final 128-dim embedding space
    )

  def forward_once(self, x):
    # Extract spatial features
    out = self.cnn(x)
    out = out.view(out.size(0), -1)
    out = self.fc(out)

    # CRITICAL: L2 Normalize embeddings to prevent model collapse
    out = F.normalize(out, p=2, dim=1)
    return out

  def forward(self, input1, input2):
    output1 = self.forward_once(input1)
    output2 = self.forward_once(input2)
    return output1, output2