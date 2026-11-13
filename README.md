# Biochemical Virtual Screening via Siamese Neural Networks

An end-to-end deep learning pipeline that uses a **Siamese Convolutional Neural Network (CNN)** and **Contrastive Loss** to perform embedding-based molecular similarity matching and virtual screening. The system converts chemical structure text descriptions (**SMILES strings**) into 2D molecular structure image tensors using **RDKit**, applying L2-normalised continuous embeddings to prevent model collapse and enable robust metric learning across biochemical datasets.

---

## Architecture Overview

The pipeline relies on a **Siamese Neural Network** design, which processes pairs of molecular inputs through two identical twin subnetworks that share weights:

*   **Twin CNN Branches:** Two parallel convolutional networks process the 2D molecular structure images (generated via RDKit) to extract high-level spatial and topological features (such as ring systems, functional groups, and branch points).
*   **Shared Embedding Space with L2 Normalisation (x):** Both branches project their respective input molecules into a normalised, low-dimensional continuous embedding vector space, preventing zero-vector collapse.
*   **Contrastive Loss Function:** Rather than traditional cross-entropy classification, the model optimises using Contrastive Loss. It explicitly pulls structurally or biologically similar molecules close together in the embedding space while pushing dissimilar compound classes apart based on a defined margin.

---

## Core Capabilities

*   **Automated SMILES-to-Tensor Conversion:** Seamlessly translates raw chemical line notations (SMILES) into normalised 2D structural image tensors using RDKit rendering pipelines.
*   **Scalable Dataset Ingestion:** Built-in support for benchmark cheminformatics datasets (such as the MoleculeNet BBBP dataset) via synchronised schema mapping (`num`, `name`, `p_np`, `smiles`).
*   **Metric-Based One-Shot Screening & CSV Reporting:** Enables rapid querying of a reference drug candidate against thousands of compound libraries using fast Euclidean distance calculations, automatically exporting timestamped hit-ranking reports.
*   **Code Quality & Linting via Ruff:** Integrates **Ruff** for automated linting, import sorting, and code formatting to maintain strict Python style and code quality standards.
*   **Containerised Deployment:** Fully reproducible environment packaged via Docker for seamless CI/CD and production deployment.

---

## Practical Use Cases

*   **Lead Compound Optimisation & Analog Discovery:** Quickly screen large compound libraries to identify structural analogs and derivatives closest to a known active reference drug (e.g., finding alternatives to Aspirin or Ibuprofen).
*   **Target-Specific Virtual Screening:** Filter out inactive candidates early in the drug discovery pipeline before investing in expensive in-vitro biological assays.
*   **Molecular Property & Toxicity Prediction:** Classify compounds based on functional characteristics or biological barriers (such as Blood-Brain Barrier Penetration) using binary activity labels.

---

## Project Structure

```text
biochem_siamese_project/
│
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD pipeline
├── data/                       
│   └── molecules.csv           # Benchmark dataset (e.g., BBBP dataset)
├── models/                     
│   ├── __init__.py
│   └── siamese_net.py          # Siamese CNN twin network architecture with L2 norm
├── utils/                      
│   ├── __init__.py
│   ├── dataset.py              # Siamese paired dataset builder
│   ├── metrics.py              # Contrastive loss implementation
│   └── rdkit_loader.py         # RDKit SMILES-to-tensor processor & synchronised CSV loader
│
├── Dockerfile                  # Containerisation setup with RDKit dependencies & model weights
├── train.py                    # Model training script
├── evaluate.py                 # Virtual screening, hit ranking & CSV report generator
└── requirements.txt            # Project dependencies

```

---

## Prerequisites & Installation

Ensure you have Python installed along with a virtual environment. The core dependencies include **PyTorch**, **TorchVision**, **RDKit**, and **Ruff**.

1. **Clone or navigate to your project directory:**

```bash
cd ~/biochem_siamese_project

```

2. **Activate your virtual environment:**

```bash
source .venv/bin/activate

```

3. **Install dependencies (including Ruff):**

```bash
pip install -r requirements.txt

```

4. **Run code checks / linter:**

```bash
ruff check .

```

---

## Dataset Setup

This project uses the public **BBBP (Blood-Brain Barrier Penetration)** dataset from MoleculeNet as its benchmark training data.

1. Create a `data` directory and download the dataset CSV:

```bash
mkdir -p data
curl -o data/molecules.csv https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv

```

2. The dataset loader automatically parses the standard BBBP schema (`num`, `name`, `p_np`, `smiles`).



---

## Usage

### 1. Train the Siamese Network

Train the Siamese twin network on the molecular dataset using Contrastive Loss. The script automatically loads SMILES strings via RDKit, generates structured image tensors, forms similarity/dissimilarity pairs, and saves the trained weights to `siamese_biochem_model.pth`.

```bash
python train.py

```

### 2. Run Virtual Screening (Local Evaluation)

Perform one-shot virtual screening by querying a reference molecule against a library of candidates. The script outputs top hits and automatically exports a timestamped CSV report (e.g., `screening_report_Ibuprofen_*.csv`).

```bash
python evaluate.py --name "Ibuprofen" --smiles "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" --top_k 10

```

### 3. Run Virtual Screening via Docker

Build and run the containerised pipeline to perform screening inside an isolated, reproducible environment:

```bash
# Build the Docker image
docker build -t biochem-siamese .

# Run default screening (Aspirin)
docker run --rm biochem-siamese

# Run screening with custom arguments (Ibuprofen)
docker run --rm biochem-siamese python evaluate.py --name "Ibuprofen" --smiles "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" --top_k 10

```

---

## License

This project is open-source and available under the terms of the MIT License.
