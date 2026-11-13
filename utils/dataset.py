import random

import torch
from torch.utils.data import Dataset


class SiameseBiochemDataset(Dataset):
  """Custom Dataset for Siamese Network training using molecular pairs.

  Generates pairs of molecular image tensors and binary labels (0 for similar
  class, 1 for dissimilar class) for Contrastive Loss optimization.
  """

  def __init__(self, df, transform=None):
    self.df = df
    self.transform = transform
    self.all_labels = self.df["p_np"].values
    self.smiles_list = self.df["smiles"].values

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    # Anchor molecule
    img1 = self.df.iloc[index]["tensor"]
    label1 = self.df.iloc[index]["p_np"]

    # Decide whether to sample a positive pair (same class) or negative pair (different class)
    should_get_same_class = random.randint(0, 1)

    if should_get_same_class:
      same_class_indices = self.df[self.df["p_np"] == label1].index.tolist()
      idx2 = random.choice(same_class_indices)
    else:
      # Linter fix: changed ambiguous variable 'l' to 'lbl'
      different_labels = [lbl for lbl in self.all_labels if lbl != label1]
      if not different_labels:
        idx2 = index
      else:
        diff_label = random.choice(different_labels)
        diff_class_indices = self.df[
            self.df["p_np"] == diff_label
        ].index.tolist()
        idx2 = random.choice(diff_class_indices)

    img2 = self.df.iloc[idx2]["tensor"]
    label2 = self.df.iloc[idx2]["p_np"]

    # Apply optional transforms
    if self.transform:
      img1 = self.transform(img1)
      img2 = self.transform(img2)

    # Contrastive loss target: 0 if same class, 1 if different class
    target = 0.0 if label1 == label2 else 1.0

    return img1, img2, torch.tensor(target, dtype=torch.float32)