import torch
import pytest
from models.siamese_net import SiameseNetwork
from utils.metrics import ContrastiveLoss
from utils.dataset import SiameseBiochemDataset
from utils.rdkit_loader import smiles_to_tensor


def test_siamese_network_forward():
  """Test forward pass, output shape, and L2 normalization of the Siamese network."""
  model = SiameseNetwork()
  model.eval()

  # Create dummy input batches: (Batch Size=4, Channels=3, Height=128, Width=128)
  input1 = torch.randn(4, 3, 128, 128)
  input2 = torch.randn(4, 3, 128, 128)

  with torch.no_grad():
    out1, out2 = model(input1, input2)

  # Verify embeddings match expected 128-dimensional space
  assert out1.shape == (4, 128)
  assert out2.shape == (4, 128)

  # Verify L2 normalization (norms should be close to 1.0)
  norms1 = torch.norm(out1, p=2, dim=1)
  norms2 = torch.norm(out2, p=2, dim=1)
  assert torch.allclose(norms1, torch.ones_like(norms1), atol=1e-5)
  assert torch.allclose(norms2, torch.ones_like(norms2), atol=1e-5)


def test_contrastive_loss():
  """Test contrastive loss behavior for similar and dissimilar pairs."""
  criterion = ContrastiveLoss(margin=1.0)

  # Identical embeddings with similar label (0.0) -> Loss should be 0
  out1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
  out2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
  labels_similar = torch.tensor([0.0, 0.0])
  loss_similar = criterion(out1, out2, labels_similar)
  assert torch.allclose(loss_similar, torch.tensor(0.0), atol=1e-5)

  # Identical embeddings with dissimilar label (1.0) -> Loss should penalize closeness
  labels_dissimilar = torch.tensor([1.0, 1.0])
  loss_dissimilar = criterion(out1, out2, labels_dissimilar)
  assert loss_dissimilar.item() > 0.0


def test_siamese_biochem_dataset():
  """Test dataset pairing length, tensor dimensions, and target types."""
  dummy_tensors = torch.randn(10, 3, 128, 128)
  dummy_labels = torch.tensor([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])

  dataset = SiameseBiochemDataset(dummy_tensors, dummy_labels)
  assert len(dataset) == 10

  img1, img2, target = dataset[0]
  assert img1.shape == (3, 128, 128)
  assert img2.shape == (3, 128, 128)
  assert isinstance(target, torch.Tensor)
  assert target.item() in [0.0, 1.0]


def test_smiles_to_tensor_conversion():
  """Test RDKit SMILES parsing for both valid molecules and error fallbacks."""
  # Valid SMILES (Aspirin)
  tensor_valid = smiles_to_tensor("CC(=O)OC1=CC=CC=C1C(=O)O", image_size=(64, 64))
  assert tensor_valid.shape == (3, 64, 64)
  assert isinstance(tensor_valid, torch.Tensor)

  # Invalid SMILES should safely return a fallback image tensor
  tensor_invalid = smiles_to_tensor("INVALID_SMILES", image_size=(64, 64))
  assert tensor_invalid.shape == (3, 64, 64)
  assert isinstance(tensor_invalid, torch.Tensor)
