import random
import torch
from torch.utils.data import Dataset


class SiameseBiochemDataset(Dataset):
  """Custom Dataset for Siamese Network training using molecular pairs.

  Generates pairs of molecular image tensors and binary labels (0.0 for similar
  class, 1.0 for dissimilar class) for Contrastive Loss optimization.
  """

  def __init__(self, molecule_dataset, labels, transform=None):
    self.tensors = molecule_dataset
    # Convert tensor labels to plain Python ints/floats for fast O(1) lookup
    self.labels = (
        labels.tolist() if isinstance(labels, torch.Tensor) else list(labels)
    )
    self.transform = transform

    # Pre-compute positional indices per class once in __init__
    self.class_to_indices = {}
    for pos_idx, lbl in enumerate(self.labels):
      self.class_to_indices.setdefault(lbl, []).append(pos_idx)
    self.unique_labels = list(self.class_to_indices.keys())

  def __len__(self):
    return len(self.tensors)

  def __getitem__(self, index):
    img1 = self.tensors[index]
    label1 = self.labels[index]

    should_get_same_class = random.randint(0, 1)

    if should_get_same_class and len(self.class_to_indices[label1]) > 1:
      # Exclude self (index) so pairs are always two distinct molecules
      candidates = [
          i for i in self.class_to_indices[label1] if i != index
      ]
      idx2 = random.choice(candidates)
    else:
      diff_labels = [l for l in self.unique_labels if l != label1]
      if not diff_labels:
        idx2 = index
      else:
        diff_label = random.choice(diff_labels)
        idx2 = random.choice(self.class_to_indices[diff_label])

    img2 = self.tensors[idx2]
    label2 = self.labels[idx2]

    if self.transform:
      img1 = self.transform(img1)
      img2 = self.transform(img2)

    # 0.0 = same class (similar), 1.0 = different class (dissimilar)
    target = 0.0 if label1 == label2 else 1.0
    return img1, img2, torch.tensor(target, dtype=torch.float32)