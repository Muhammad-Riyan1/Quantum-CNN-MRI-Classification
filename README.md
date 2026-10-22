# Quantum CNN for MRI Image Classification

## Overview
This project implements a Quantum Convolutional Neural Network (QCNN) using PyTorch and PennyLane for MRI image classification. The model combines classical convolutional neural network layers with a quantum circuit to extract features and classify MRI images.

## Technologies Used
- Python
- PyTorch
- PennyLane
- Torchvision
- NumPy
- Matplotlib
- Seaborn
- Pandas
- Scikit-learn

## Model Architecture
The model consists of:
1. Image preprocessing and resizing to 64 × 64 pixels.
2. Classical CNN layers for feature extraction.
3. Fully connected layers for feature reduction.
4. A quantum layer with 4 qubits and 2 trainable quantum layers.
5. An output layer for classification.

## Dataset
The model loads training and testing images using PyTorch's `ImageFolder`.

Expected dataset structure:

```text
MRI/
├── Training/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── Testing/
    ├── class_1/
    ├── class_2/
    └── ...
```

Update the dataset paths in `QCNN.py` to match your local environment.

## Installation

Install the required libraries:

```bash
pip install torch torchvision pennylane matplotlib seaborn pandas scikit-learn
```

## Usage
1. Download or clone this repository.
2. Prepare the MRI dataset in the required directory structure.
3. Update the dataset paths in `QCNN.py`.
4. Run the model:

```bash
python QCNN.py
```

The script trains the model, evaluates its performance, displays performance graphs, and saves the model weights and training results.

## Output Files
- `quantum_mri_cnn.pth` — saved model weights.
- `quantum_training_results.csv` — training and validation results.

## Disclaimer
This project is intended for research and educational purposes. It is not a clinically validated medical diagnostic system.

## Author
Add your name here.
