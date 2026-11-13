# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for RDKit and compiling packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxrender1 \
    libxext6 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies (PyTorch, RDKit, pandas, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files, including models, utils, scripts, data, and trained weights (.pth)
COPY . .

# Set the default command to run virtual screening using Aspirin as the default reference molecule
CMD ["python", "evaluate.py"]