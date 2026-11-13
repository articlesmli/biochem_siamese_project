import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.siamese_net import SiameseNetwork
from utils.dataset import SiameseBiochemDataset
from utils.metrics import ContrastiveLoss
from utils.rdkit_loader import load_real_biochem_data


def train_model():
  # Setup device (GPU if available, else CPU)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Training on device: {device}")

  # 1. Load Real Biochemistry Data using RDKit
  print("Loading real molecular data via RDKit...")
  molecule_tensors, label_tensors = load_real_biochem_data()

  # Create Siamese paired dataset from the 2,039 BBBP molecules
  dataset = SiameseBiochemDataset(
      molecule_dataset=molecule_tensors, labels=label_tensors
  )
  # Use a larger batch size (64) for efficient gradient updates
  dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

  # 2. Initialize Model, Loss, and Optimizer
  model = SiameseNetwork().to(device)
  criterion = ContrastiveLoss(
      margin=1.0
  )  # Standard margin for normalized embeddings
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  # 3. Training Loop
  num_epochs = 15  # 15 epochs is plenty for 2,000+ molecules with batch_size=64
  model.train()

  for epoch in range(num_epochs):
    epoch_loss = 0.0

    for mol1, mol2, labels in dataloader:
      mol1 = mol1.to(device)
      mol2 = mol2.to(device)
      labels = labels.to(device)

      # Zero gradients
      optimizer.zero_grad()

      # Forward pass: get embeddings for both molecule branches
      output1, output2 = model(mol1, mol2)

      # Compute contrastive loss
      loss = criterion(output1, output2, labels)

      # Backward pass and optimization
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Contrastive Loss: {avg_loss:.4f}"
    )

  # Save trained model weights
  torch.save(model.state_dict(), "siamese_biochem_model.pth")
  print("Training complete! Model saved to 'siamese_biochem_model.pth'.")


if __name__ == "__main__":
  train_model()
