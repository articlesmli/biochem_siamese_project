import csv
from rdkit import Chem
from rdkit.Chem import Draw
import torch
import torchvision.transforms as transforms


def smiles_to_tensor(smiles_string, image_size=(128, 128)):
  """Converts a chemical SMILES string into a PyTorch image tensor

  representing its 2D molecular structure.
  """
  mol = Chem.MolFromSmiles(smiles_string)
  if mol is None:
    return None  # Skip invalid SMILES gracefully

  img = Draw.MolToImage(mol, size=image_size)
  transform = transforms.Compose([transforms.ToTensor()])
  return transform(img)


def load_real_biochem_data(csv_filepath="data/molecules.csv"):
  """Loads real molecular data from the BBBP CSV file

  using columns: num, name, p_np, smiles
  """
  molecule_tensors = []
  labels = []

  print(f"Loading molecular data from {csv_filepath}...")

  with open(csv_filepath, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
      smiles = row.get("smiles")
      p_np_val = row.get("p_np")

      if not smiles or p_np_val == "" or p_np_val is None:
        continue

      try:
        # Convert label to integer (handles potential float formats like '1.0')
        label = int(float(p_np_val))
      except ValueError:
        continue

      # Convert SMILES to image tensor via RDKit
      tensor = smiles_to_tensor(smiles)
      if tensor is not None:
        molecule_tensors.append(tensor)
        labels.append(label)

  if not molecule_tensors:
    raise ValueError("No valid molecules were loaded from the CSV file!")

  molecule_tensors = torch.stack(molecule_tensors)
  label_tensors = torch.tensor(labels, dtype=torch.long)

  print(
      f"Successfully loaded {len(molecule_tensors)} valid molecules across"
      f" {len(set(labels))} classes from BBBP dataset."
  )
  return molecule_tensors, label_tensors