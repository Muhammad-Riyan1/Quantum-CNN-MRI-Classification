# 🧠 Quantum CNN for MRI Image Classification

<p align="center">
  <b>Hybrid Classical–Quantum Deep Learning for MRI Image Analysis</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/PennyLane-Quantum%20Computing-6C43E0" alt="PennyLane">
  <img src="https://img.shields.io/badge/Status-Research%20Project-orange" alt="Research Project">
</p>

---

## 📌 Overview

This project implements a **Hybrid Quantum Convolutional Neural Network (Quantum CNN)** for MRI image classification by combining classical deep learning with quantum computing.

The architecture uses convolutional neural network layers to extract image features and a parameterized quantum circuit to process a compact feature representation before final classification.

The model is implemented using **PyTorch** and **PennyLane** and includes training, validation, testing, performance evaluation, and visualization.

## 🎯 Key Features

- 🧠 Classical CNN-based MRI feature extraction
- ⚛️ Hybrid quantum-classical neural network architecture
- 🔬 Parameterized quantum circuit with 4 qubits
- 🔗 Trainable quantum layers using PennyLane
- 📊 Training and validation loss visualization
- 📈 Training and validation accuracy visualization
- 📋 Accuracy, precision, recall, and F1-score evaluation
- 🔥 Confusion matrix visualization
- 💾 Model weights and training results saved to files

## 🏗️ Model Architecture

The model follows a hybrid processing pipeline:

```text
                 ┌─────────────────────────┐
                 │      MRI Image Input    │
                 │       (64 × 64)         │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │    Image Preprocessing  │
                 │ Grayscale + Normalize   │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   Convolution Layer 1   │
                 │   32 Feature Channels   │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   Convolution Layer 2   │
                 │   64 Feature Channels   │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │     Max Pooling         │
                 │   Feature Reduction     │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │  Fully Connected Layer  │
                 │      128 Neurons        │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   Feature Compression   │
                 │       4 Features        │
                 └────────────┬────────────┘
                              │
                              ▼
            ╔══════════════════════════════════╗
            ║       QUANTUM PROCESSING         ║
            ║                                  ║
            ║  • 4 Qubits                      ║
            ║  • RY Data Encoding              ║
            ║  • Entangling Quantum Layers     ║
            ║  • Pauli-Z Measurements          ║
            ╚════════════════╤═════════════════╝
                             │
                             ▼
                 ┌─────────────────────────┐
                 │   Classical Output      │
                 │     Classification      │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   Predicted Class       │
                 │     MRI Image           │
                 └─────────────────────────┘
```

## ⚛️ Quantum Circuit Configuration

| Parameter | Value |
|---|---|
| Quantum framework | PennyLane |
| Quantum simulator | `default.qubit` |
| Number of qubits | 4 |
| Quantum circuit depth | 2 trainable layers |
| Data encoding | RY rotation gates |
| Entanglement | StronglyEntanglingLayers |
| Measurement | Pauli-Z expectation values |
| Quantum layer integration | `qml.qnn.TorchLayer` |

The classical network reduces the extracted image representation to four features. These features are transformed with `tanh` and passed into the quantum layer.

The quantum circuit returns four expectation values, which are processed by the final classical linear layer to produce class logits.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Neural network training |
| Torchvision | Image loading and preprocessing |
| PennyLane | Quantum circuit simulation |
| NumPy | Numerical computation |
| Matplotlib | Graph generation |
| Seaborn | Confusion matrix visualization |
| Pandas | Training results and CSV export |
| Scikit-learn | Classification metrics |

## 📂 Project Structure

```text
Quantum-CNN-MRI-Classification/
│
├── QCNN.py
│
├── README.md
│
├── requirements.txt
│
├── quantum_training_results.csv
│
└── quantum_mri_cnn.pth
```

**Note:** The CSV and `.pth` files are generated when the training script runs. They do not need to exist in the repository before running the code.

## 📊 Dataset Organization

The script uses PyTorch's `ImageFolder` to load the MRI images.

Expected directory structure:

```text
MRI/
│
├── Training/
│   ├── Class_1/
│   ├── Class_2/
│   └── ...
│
└── Testing/
    ├── Class_1/
    ├── Class_2/
    └── ...
```

Each class folder should contain images belonging to that class.

Update the dataset paths in `QCNN.py` to match your own environment. The example paths in the original code point to Google Drive in Colab.

> **Dataset notice:** The MRI dataset is not included in this repository. Use a dataset you have permission to access and share. Do not publish private patient information.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Quantum-CNN-MRI-Classification.git
cd Quantum-CNN-MRI-Classification
```

### 2. Install dependencies

```bash
pip install torch torchvision pennylane matplotlib seaborn pandas scikit-learn
```

### 3. Configure the dataset

Open `QCNN.py` and update the training and testing dataset paths.

### 4. Run the model

```bash
python QCNN.py
```

The script trains the network for 20 epochs, evaluates it on the test dataset, generates plots, and saves the training results and model weights.

## 📉 Evaluation Metrics

The model reports the following evaluation metrics:

- **Accuracy:** Overall proportion of correctly classified images.
- **Precision:** Weighted precision across the classes.
- **Recall:** Weighted recall across the classes.
- **F1-score:** Weighted harmonic mean of precision and recall.

It also generates:

- Training vs. validation loss graph
- Training vs. validation accuracy graph
- Classification performance bar chart
- Confusion matrix
- Training results table

Actual performance values should be added after running the model. No accuracy or clinical performance is claimed in advance.

## 💾 Generated Outputs

| File | Description |
|---|---|
| `quantum_mri_cnn.pth` | Trained model state dictionary |
| `quantum_training_results.csv` | Per-epoch training and validation statistics |

The model weights can be loaded later using the same model architecture and compatible configuration.

## 🔭 Future Improvements

- Compare the hybrid model with a classical CNN baseline.
- Evaluate different numbers of qubits and quantum circuit depths.
- Add reproducible random seeds and experiment configuration.
- Track per-class performance and class imbalance.
- Improve model checkpointing and inference support.
- Evaluate generalization on independent datasets.

## ⚠️ Disclaimer

This project is intended for **research and educational purposes only**. It is not a clinically validated diagnostic system and must not be used as a substitute for professional medical evaluation.


<p align="center">
  ⭐ If you find this project useful, consider starring the repository!
</p>
