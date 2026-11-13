import argparse
import csv
from datetime import datetime

import torch
import torch.nn.functional as F

from models.siamese_net import SiameseNetwork
from utils.rdkit_loader import smiles_to_tensor


def parse_arguments():
  parser = argparse.ArgumentParser(
      description=(
          "Perform virtual screening against the BBBP dataset using a trained"
          " Siamese network."
      )
  )
  parser.add_argument(
      "--smiles",
      type=str,
      default="CC(=O)OC1=CC=CC=C1C(=O)O",
      help="SMILES string of the reference molecule (default: Aspirin).",
  )
  parser.add_argument(
      "--name",
      type=str,
      default="Aspirin",
      help="Name of the reference molecule (default: Aspirin).",
  )
  parser.add_argument(
      "--top_k",
      type=int,
      default=5,
      help="Number of top virtual screening hits to display and save (default: 5).",
  )
  return parser.parse_args()


def load_candidates_synchronized(csv_filepath="data/molecules.csv"):
  """Loads candidate tensors and metadata simultaneously

  to guarantee strict 100% index alignment.
  """
  molecule_tensors = []
  valid_candidates = []

  print(f"Loading and aligning candidate library from '{csv_filepath}'...")

  with open(csv_filepath, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
      smiles = row.get("smiles")
      name = row.get("name", "Unknown")
      p_np = row.get("p_np")

      if not smiles or p_np == "" or p_np is None:
        continue

      try:
        label = int(float(p_np))
      except ValueError:
        continue

      # Convert SMILES to tensor
      tensor = smiles_to_tensor(smiles)
      if tensor is not None:
        molecule_tensors.append(tensor)
        valid_candidates.append({"name": name, "smiles": smiles, "p_np": label})

  if not molecule_tensors:
    raise ValueError("No valid molecules were loaded from the CSV file!")

  molecule_tensors = torch.stack(molecule_tensors)
  return molecule_tensors, valid_candidates


def perform_one_shot_screening():
  args = parse_arguments()

  # 1. Set device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Evaluating on device: {device}")

  # 2. Load the trained Siamese model weights
  model = SiameseNetwork().to(device)
  try:
    model.load_state_dict(
        torch.load("siamese_biochem_model.pth", map_location=device)
    )
  except FileNotFoundError:
    print(
        "Error: Model weights not found! Please run 'python train.py' first."
    )
    return

  model.eval()

  # 3. Process Reference Molecule
  print(f"Reference Molecule: {args.name} ({args.smiles})")
  try:
    reference_tensor = smiles_to_tensor(args.smiles).unsqueeze(0).to(device)
  except ValueError as e:
    print(f"Error parsing reference SMILES: {e}")
    return

  # 4. Load synchronized candidates
  try:
    candidate_tensors, valid_candidates = load_candidates_synchronized(
        "data/molecules.csv"
    )
    candidate_tensors = candidate_tensors.to(device)
  except (FileNotFoundError, ValueError, RuntimeError) as e:
    print(f"Error loading candidate library: {e}")
    return

  print(
      f"Screening reference molecule against a synchronized library of"
      f" {len(candidate_tensors)} compounds..."
  )

  with torch.no_grad():
    ref_embedding = model.forward_once(reference_tensor)
    candidate_embeddings = model.forward_once(candidate_tensors)

    ref_expanded = ref_embedding.expand_as(candidate_embeddings)
    distances = F.pairwise_distance(ref_expanded, candidate_embeddings)

    sorted_distances, sorted_indices = torch.sort(distances)

  # 5. Output top K hits and save to CSV report
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  safe_name = "".join(c if c.isalnum() else "_" for c in args.name)
  report_filename = f"screening_report_{safe_name}_{timestamp}.csv"

  print(
      f"\n--- Top {args.top_k} Virtual Screening Hits (Reference:"
      f" {args.name}) ---"
  )
  
  report_rows = []
  for rank in range(min(args.top_k, len(candidate_tensors))):
    idx = sorted_indices[rank].item()
    dist = sorted_distances[rank].item()

    hit_info = valid_candidates[idx]
    print(
        f"  Hit #{rank + 1}: {hit_info['name']} (BBBP Class: {hit_info['p_np']})"
        f" | Distance: {dist:.4f}"
    )

    report_rows.append({
        "Rank": rank + 1,
        "Reference_Name": args.name,
        "Reference_SMILES": args.smiles,
        "Hit_Name": hit_info["name"],
        "Hit_SMILES": hit_info["smiles"],
        "BBBP_Class": hit_info["p_np"],
        "Embedding_Distance": f"{dist:.4f}"
    })

  # Write report to CSV
  csv_columns = [
      "Rank",
      "Reference_Name",
      "Reference_SMILES",
      "Hit_Name",
      "Hit_SMILES",
      "BBBP_Class",
      "Embedding_Distance",
  ]
  with open(report_filename, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    writer.writeheader()
    for row in report_rows:
      writer.writerow(row)

  print(f"\n📁 Screening report successfully saved to: '{report_filename}'")


if __name__ == "__main__":
  perform_one_shot_screening()