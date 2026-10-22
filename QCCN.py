# ============================================================
# GOOGLE DRIVE MOUNT
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================================
# INSTALL REQUIRED LIBRARIES
# ============================================================
#!pip install pennylane
# ============================================================
# IMPORT LIBRARIES
# ============================================================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import pennylane as qml
from pennylane import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# DEVICE CONFIGURATION
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# ============================================================
# GRAPH STYLE FOR RESEARCH PAPER
# ============================================================
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.4)

# ============================================================
# IMAGE TRANSFORMS(Every MRI image goes through:)
# ============================================================
transform = transforms.Compose([
    transforms.Resize((64, 64)), #(64 × 64)
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),#(Converts image into numerical tensor.)
    transforms.Normalize(mean=[0.5], std=[0.5]) #(Pixel values are scaled roughly between:-1 and +1
    #This improves training stability.)
])

# ============================================================
# LOAD DATASETS
# ============================================================
full_dataset = datasets.ImageFolder(
    root="/content/drive/MyDrive/MRI/Training",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root="/content/drive/MyDrive/MRI/Testing",
    transform=transform
)

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size]
)

# ============================================================
# DATA LOADERS
# ============================================================
batch_size = 16 #(Each training step processes 16 images at a time.)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# ============================================================
# CLASS NAMES
# ============================================================
class_names = full_dataset.classes

print("Classes:", class_names)
print("Total Classes:", len(class_names))

# ============================================================
# QUANTUM CONFIGURATION
# ============================================================
#Number of Qubits
n_qubits = 4 #4 Quantum Bits (will represent extracted MRI features.)

#Number of trainable quantum layers.More quantum expressiveness and More computation
q_depth = 2

dev = qml.device("default.qubit", wires=n_qubits)

# ============================================================
# QUANTUM CIRCUIT -->Defines the quantum neural network.
# ============================================================
@qml.qnode(dev, interface="torch")

def quantum_circuit(inputs, weights):

    # Data Encoding (MRI features are encoded into qubit rotations , Feature 1 → Qubit 1 , Feature 2 → Qubit 2)
    for i in range(n_qubits):
        qml.RY(inputs[:, i], wires=i)#Rotation gate: stores information inside quantum states.

    # Trainable Quantum Layers
    qml.templates.StronglyEntanglingLayers(
        weights,
        wires=range(n_qubits)
    )

    # Measurements
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)] # Measures each qubit --> return 4 values between:-1 and +1

# ============================================================
# QUANTUM WEIGHT SHAPES
# ============================================================
weight_shapes = {
    "weights": (q_depth, n_qubits, 3)
}

# ============================================================
# QUANTUM LAYER
# ============================================================
quantum_layer = qml.qnn.TorchLayer( #Converts the quantum circuit into a PyTorch layer.
    quantum_circuit,
    weight_shapes
)

# ============================================================
# QUANTUM CNN MODEL
# ============================================================
class QuantumCNN(nn.Module):

    def __init__(self, num_classes):

        super(QuantumCNN, self).__init__()

        # CNN 1st Layers

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(32)
        #layer 2 Extracts higher-level MRI patterns.
        self.conv2 = nn.Conv2d(#64 x 64 -->32x32
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)#32×32 → 16×16


        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 16 * 16, 128) #(16384,128) Compresses features.

        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, n_qubits) #(128,4)

        # Quantum Layer
        self.quantum = quantum_layer

        # Output Layer
        self.fc3 = nn.Linear(n_qubits, num_classes) #64 × 16 × 16

    def forward(self, x):

        # CNN Feature Extraction
        x = self.pool(
            F.relu(self.bn1(self.conv1(x)))
        )

        x = self.pool(
            F.relu(self.bn2(self.conv2(x)))
        )

        # Flatten
        x = x.view(x.size(0), -1)

        # Dense Layers 16x16x16 = 16384 Features
        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        # Normalize for Quantum Circuit
        x = torch.tanh(x)

        # Move tensor to CPU for quantum layer
        x_cpu = x.cpu()

        # Quantum Layer
        x = self.quantum(x_cpu).to(device)

        # Final Output
        x = self.fc3(x)

        return x

# ============================================================
# MODEL INITIALIZATION
# ============================================================
num_classes = len(class_names)

model = QuantumCNN(num_classes=num_classes).to(device)

print(model)

# ============================================================
# LOSS FUNCTION & OPTIMIZER
# ============================================================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)

# ============================================================
# STORAGE FOR RESULTS
# ============================================================
train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_model(model, train_loader, val_loader,
                criterion, optimizer, epochs=20):

    for epoch in range(epochs):

        # ====================================================
        # TRAINING
        # ====================================================
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward Pass
            outputs = model(images)

            loss = criterion(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # Statistics
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)

        train_acc = 100 * correct / total

        # ====================================================
        # VALIDATION
        # ====================================================
        model.eval()

        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)

                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_loader)

        val_acc = 100 * val_correct / val_total

        # ====================================================
        # STORE RESULTS
        # ====================================================
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # ====================================================
        # PRINT RESULTS
        # ====================================================
        print("=" * 60)

        print(f"Epoch [{epoch+1}/{epochs}]")

        print(f"Train Loss      : {train_loss:.4f}")
        print(f"Validation Loss : {val_loss:.4f}")

        print(f"Train Accuracy  : {train_acc:.2f}%")
        print(f"Validation Acc  : {val_acc:.2f}%")

        print("=" * 60)

# ============================================================
# EVALUATION FUNCTION
# ============================================================
def evaluate_model(model, test_loader):

    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())

            all_predictions.extend(predicted.cpu().numpy())

    # ========================================================
    # PERFORMANCE METRICS
    # ========================================================
    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average='weighted'
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average='weighted'
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average='weighted'
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================
    print("\n")
    print("=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(f"Accuracy  : {accuracy*100:.2f}%")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("=" * 60)

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================
    print("\nClassification Report:\n")

    print(classification_report(
        all_labels,
        all_predictions,
        target_names=class_names
    ))

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================
    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='mako',
        linewidths=1,
        linecolor='white',
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(
        "Confusion Matrix",
        fontsize=18,
        fontweight='bold'
    )

    plt.xlabel(
        "Predicted Labels",
        fontsize=14
    )

    plt.ylabel(
        "True Labels",
        fontsize=14
    )

    plt.xticks(rotation=45, fontsize=11)

    plt.yticks(rotation=0, fontsize=11)

    plt.tight_layout()

    plt.show()

    return accuracy, precision, recall, f1

# ============================================================
# TRAINING
# ============================================================
train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs=20
)

# ============================================================
# TESTING
# ============================================================
accuracy, precision, recall, f1 = evaluate_model(
    model,
    test_loader
)

# ============================================================
# TRAINING VS VALIDATION LOSS GRAPH
# ============================================================
plt.figure(figsize=(8, 5))

plt.plot(
    train_losses,
    marker='o',
    linewidth=3,
    label='Training Loss'
)

plt.plot(
    val_losses,
    marker='s',
    linewidth=3,
    label='Validation Loss'
)

plt.title(
    "Training vs Validation Loss",
    fontsize=16,
    fontweight='bold'
)

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# ============================================================
# TRAINING VS VALIDATION ACCURACY GRAPH
# ============================================================
plt.figure(figsize=(8, 5))

plt.plot(
    train_accuracies,
    marker='o',
    linewidth=3,
    label='Training Accuracy'
)

plt.plot(
    val_accuracies,
    marker='s',
    linewidth=3,
    label='Validation Accuracy'
)

plt.title(
    "Training vs Validation Accuracy",
    fontsize=16,
    fontweight='bold'
)

plt.xlabel("Epochs")

plt.ylabel("Accuracy (%)")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# ============================================================
# PERFORMANCE METRICS BAR GRAPH
# ============================================================
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

values = [
    accuracy * 100,
    precision * 100,
    recall * 100,
    f1 * 100
]

plt.figure(figsize=(8, 5))

bars = plt.bar(metrics, values)

plt.title(
    "Performance Metrics",
    fontsize=16,
    fontweight='bold'
)

plt.ylabel("Percentage (%)")

plt.ylim(0, 100)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 1,
        f'{height:.2f}',
        ha='center',
        fontsize=11
    )

plt.tight_layout()

plt.show()

# ============================================================
# RESULTS TABLE
# ============================================================
results_table = pd.DataFrame({

    "Epoch": list(range(1, 21)),

    "Train Loss": train_losses,

    "Validation Loss": val_losses,

    "Train Accuracy": train_accuracies,

    "Validation Accuracy": val_accuracies
})

print("\n")
print("=" * 70)
print("TRAINING RESULTS TABLE")
print("=" * 70)

print(results_table)

# ============================================================
# SAVE RESULTS TABLE
# ============================================================
results_table.to_csv(
    "quantum_training_results.csv",
    index=False
)

# ============================================================
# SAVE MODEL
# ============================================================
torch.save(
    model.state_dict(),
    "quantum_mri_cnn.pth"
)

print("\nQuantum CNN Model Saved Successfully!")