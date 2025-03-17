import os
import shutil
import random
from collections import defaultdict

# Define paths
DATASET_PATH = "/Users/mac/Desktop/datasets/malimg_dataset"  # Update this to your dataset path
NON_IID_PATH = "/Users/mac/Desktop/datasets/noniid"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR = os.path.join(DATASET_PATH, "val")
TEST_DIR = os.path.join(DATASET_PATH, "test")

# Number of clients
num_clients = 5

# Get malware families (Ignore non-directory files like .DS_Store)
malware_families = [f for f in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, f))]

# Create Non-IID directory
os.makedirs(NON_IID_PATH, exist_ok=True)

# Assign malware families to clients (Label Skew)
client_data = defaultdict(list)
for family in malware_families:
    client_id = random.randint(0, num_clients - 1)  # Randomly assign a client
    client_data[client_id].append(family)

# Function to distribute files
def distribute_files(source_dir, client_folder, client_id):
    os.makedirs(client_folder, exist_ok=True)
    for family in os.listdir(source_dir):
        # Ignore non-directory files
        if family in client_data[client_id] and os.path.isdir(os.path.join(source_dir, family)):
            source_family_dir = os.path.join(source_dir, family)
            target_family_dir = os.path.join(client_folder, family)
            shutil.copytree(source_family_dir, target_family_dir, dirs_exist_ok=True)

# Create client datasets
for client_id in range(num_clients):
    client_folder = os.path.join(NON_IID_PATH, f"client_{client_id}")
    distribute_files(TRAIN_DIR, os.path.join(client_folder, "train"), client_id)
    distribute_files(VAL_DIR, os.path.join(client_folder, "val"), client_id)
    distribute_files(TEST_DIR, os.path.join(client_folder, "test"), client_id)

print("Non-IID dataset created successfully!")