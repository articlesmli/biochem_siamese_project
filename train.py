import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.siamese_net import SiameseNetwork
from utils.dataset import SiameseBiochemDataset
from utils.metrics import ContrastiveLoss
from utils.rdkit_loader import load_bbbp_dataset


def evaluate_val(model, val_loader, criterion, device, threshold=0.5):
  model.eval()
  total_loss = 0.0
  correct_pairs = 0
  total_pairs = 0

  with torch.no_grad():
    for mol1, mol2, labels in val_loader:
      mol1 = mol1.to(device)
      mol2 = mol2.to(device)
      labels = labels.to(device)

      output1, output2 = model(mol1, mol2)
      loss = criterion(output1, output2, labels)
      total_loss += loss.item() * labels.size(0)

      distances = F.pairwise_distance(output1, output2, keepdim=False)
      preds = (distances > threshold).float()
      correct_pairs += (preds == labels).sum().item()
      total_pairs += labels.size(0)

  return total_loss / total_pairs, correct_pairs / total_pairs


def train_model():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Training on device: {device}")

  # 1. Load Real Biochemistry Data using RDKit
  print("Loading real molecular data via RDKit...")
  records = load_bbbp_dataset("data/molecules.csv")
  molecule_tensors = torch.stack([r["tensor"] for r in records])
  label_tensors = torch.tensor([r["p_np"] for r in records], dtype=torch.float32)

  # Reproducible 80/20 train/validation split
  num_samples = len(molecule_tensors)
  generator = torch.Generator().manual_seed(42)
  perm = torch.randperm(num_samples, generator=generator)
  split_idx = int(0.8 * num_samples)

  train_idx = perm[:split_idx]
  val_idx = perm[split_idx:]

  train_dataset = SiameseBiochemDataset(
      molecule_dataset=molecule_tensors[train_idx],
      labels=label_tensors[train_idx],
  )
  val_dataset = SiameseBiochemDataset(
      molecule_dataset=molecule_tensors[val_idx],
      labels=label_tensors[val_idx],
  )

  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

  # 2. Initialize Model, Loss, and Optimizer
  margin = 1.0
  model = SiameseNetwork().to(device)
  criterion = ContrastiveLoss(margin=margin)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

  # 3. Training & Validation Loop
  num_epochs = 20
  best_val_loss = float("inf")

  for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_train = 0

    for mol1, mol2, labels in train_loader:
      mol1 = mol1.to(device)
      mol2 = mol2.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()
      output1, output2 = model(mol1, mol2)
      loss = criterion(output1, output2, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item() * labels.size(0)
      total_train += labels.size(0)

    train_loss = running_loss / total_train
    val_loss, val_acc = evaluate_val(
        model, val_loader, criterion, device, threshold=margin / 2.0
    )

    saved_flag = ""
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      torch.save(model.state_dict(), "siamese_biochem_model.pth")
      saved_flag = " [Saved Best]"

    print(
        f"Epoch [{epoch + 1:02d}/{num_epochs:02d}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Pair Acc: {val_acc:.4f}{saved_flag}"
    )

  print(
      "\nTraining complete! Best model weights saved to"
      " 'siamese_biochem_model.pth'."
  )


if __name__ == "__main__":
  train_model()