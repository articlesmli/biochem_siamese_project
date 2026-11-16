import csv

from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from torchvision import transforms


def smiles_to_tensor(smiles_string, image_size=(128, 128)):
  """Converts a SMILES string into a normalized 3-channel RGB image tensor using RDKit."""
  mol = Chem.MolFromSmiles(smiles_string)
  if mol is None:
    img = Image.new("RGB", image_size, (255, 255, 255))
  else:
    img = Draw.MolToImage(mol, size=image_size).convert("RGB")

  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
  ])
  return transform(img)


def load_bbbp_dataset(csv_path="data/molecules.csv"):
  """Loads the BBBP dataset CSV containing schema: num, name, p_np, smiles

  and generates RDKit structural image tensors for each molecule.
  """
  data = []
  with open(csv_path, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
      num = row.get("num", "")
      name = row.get("name", "")
      raw_label = row.get("p_np", "")
      if raw_label in (None, ""):
        continue
      try:
        p_np = int(float(raw_label))
      except ValueError:
        continue
      smiles = row.get("smiles", "")

      if smiles:
        tensor = smiles_to_tensor(smiles)
        data.append({
            "num": num,
            "name": name,
            "p_np": p_np,
            "smiles": smiles,
            "tensor": tensor,
        })
  return data