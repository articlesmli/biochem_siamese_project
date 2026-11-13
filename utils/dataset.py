import random
import torch
from torch.utils.data import Dataset


class SiameseBiochemDataset(Dataset):
  """Custom Dataset for pairing molecular representations and target activity classes

  for Siamese network training.
  """

  def __init__(self, molecule_dataset, labels):
    self.molecule_dataset = molecule_dataset
    self.labels = labels

    # Group sample indices by their biochemical class / target activity
    self.class_to_indices = {}
    for idx, label in enumerate(self.labels):
      lbl = label.item() if isinstance(label, torch.Tensor) else label
      if lbl not in self.class_to_indices:
        self.class_to_indices[lbl] = []
      self.class_to_indices[lbl].append(idx)

    self.all_labels = list(self.class_to_indices.keys())

  def __len__(self):
    return len(self.molecule_dataset)

  def __getitem__(self, index):
    # 1. Get anchor molecule and its class label
    mol1 = self.molecule_dataset[index]
    label1 = self.labels[index]
    label1 = (
        label1.item() if isinstance(label1, torch.Tensor) else label1
    )

    # 2. Randomly decide whether to form a similar pair (1) or dissimilar pair (0)
    should_get_same_class = random.randint(0, 1)

    if should_get_same_class and len(self.class_to_indices[label1]) > 1:
      # Sample another molecule from the EXACT same biochemical class
      same_class_indices = [
          i for i in self.class_to_indices[label1] if i != index
      ]
      idx2 = random.choice(
          same_class_indices
          if same_class_indices
          else self.class_to_indices[label1]
      )
      target = torch.tensor(1.0, dtype=torch.float32)
    else:
      # Sample a molecule from a DIFFERENT class
      different_labels = [l for l in self.all_labels if l != label1]
      if not different_labels:
        idx2 = index
      else:
        label2 = random.choice(different_labels)
        idx2 = random.choice(self.class_to_indices[label2])
      target = torch.tensor(0.0, dtype=torch.float32)

    mol2 = self.molecule_dataset[idx2]

    return mol1, mol2, target