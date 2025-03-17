import os
import shutil
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score

# 🔥 Ensure hidden MacOS files are ignored
def get_families(directory):
    return [f for f in os.listdir(directory) if not f.startswith('.')]

# 📂 Define dataset paths
DATASET_PATH = "/Users/mac/Desktop/datasets/malimg_dataset"  # Update path
NON_IID_PATH = "/Users/mac/Desktop/datasets/noniid"

TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR = os.path.join(DATASET_PATH, "val")
TEST_DIR = os.path.join(DATASET_PATH, "test")

# 📌 Number of federated clients
num_clients = 3

# 🔥 Get malware families (Ensuring `.DS_Store` is ignored)
malware_families = get_families(TRAIN_DIR)

# ✅ Dynamically determine number of classes
num_classes = len(malware_families)
print(f"✅ Number of detected classes: {num_classes}")

# 📌 Create Non-IID client partitions (Label Skew)
client_data = defaultdict(list)
for family in malware_families:
    client_id = random.randint(0, num_clients - 1)
    client_data[client_id].append(family)

# 📂 Distribute dataset to clients
def distribute_files(source_dir, client_folder, client_id):
    os.makedirs(client_folder, exist_ok=True)
    for family in os.listdir(source_dir):
        if family in client_data[client_id]:
            source_family_dir = os.path.join(source_dir, family)
            target_family_dir = os.path.join(client_folder, family)
            shutil.copytree(source_family_dir, target_family_dir, dirs_exist_ok=True)

# 🔥 Create client datasets
for client_id in range(num_clients):
    client_folder = os.path.join(NON_IID_PATH, f"client_{client_id}")
    distribute_files(TRAIN_DIR, os.path.join(client_folder, "train"), client_id)
    distribute_files(VAL_DIR, os.path.join(client_folder, "val"), client_id)
    distribute_files(TEST_DIR, os.path.join(client_folder, "test"), client_id)

print("✅ Non-IID dataset created successfully!")

# 📌 Custom Malware Dataset Loader
class MalwareDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        label = self.label_map[self.labels[index]]
        
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.image_paths)

# 🧠 Simple CNN Model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, (8, 8))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 🎯 Training function for a single client
def train_client(model, train_loader, criterion, optimizer, epochs=1):
    model.train()
    losses = []
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

    return model.state_dict(), np.mean(losses)

# 🚀 Federated Learning Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_model = CNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()

# 📌 Train each client and aggregate updates
global_weights = global_model.state_dict()
client_accuracies = []
client_losses = []

for client_id in range(num_clients):
    print(f"🚀 Training client {client_id}...")
    
    client_train_dir = os.path.join(NON_IID_PATH, f"client_{client_id}", "train")
    train_dataset = datasets.ImageFolder(client_train_dir, transform=transforms.Compose([
        transforms.Resize((64, 64)), 
        transforms.ToTensor()
    ]))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    client_model = CNN(num_classes).to(device)
    optimizer = optim.Adam(client_model.parameters(), lr=0.001)

    client_weights, client_loss = train_client(client_model, train_loader, criterion, optimizer)

    # 🔄 Aggregate weights
    global_weights = {key: (global_weights[key] + client_weights[key]) / 2 for key in global_weights}

    client_losses.append(client_loss)

# Update global model with aggregated weights
global_model.load_state_dict(global_weights)

# 🔍 Evaluate Model on Test Data
def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc, all_preds, all_labels

# 🎯 Load test data
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.ToTensor()
]))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 📌 Calculate Performance Metrics
global_accuracy, y_pred, y_true = evaluate_model(global_model, test_loader)
print(f"🌟 Global Model Accuracy: {global_accuracy:.4f}")

# 📊 Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=malware_families, yticklabels=malware_families)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Federated Learning")
plt.show()

# 📈 Plot Client Losses
plt.figure(figsize=(8, 5))
plt.plot(range(num_clients), client_losses, marker='o', linestyle='--', color='b')
plt.xlabel("Client ID")
plt.ylabel("Loss")
plt.title("Client Training Loss")
plt.grid()
plt.show()

print("✅ Federated Learning Training & Evaluation Complete!")
