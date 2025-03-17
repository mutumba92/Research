import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# Define paths (adjust these based on your dataset location)
data_dir = "/Users/mac/Desktop/datasets/MicrosoftBIG"
train_csv = os.path.join(data_dir, "train", "LargeTrain.csv")
test_csv = os.path.join(data_dir, "test", "LargeTest.csv")
labels_csv = os.path.join(data_dir, "train", "trainLabels.csv")
output_train_dir = os.path.join(data_dir, "train_images")
output_test_dir = os.path.join(data_dir, "test_images")

# Ensure output directories exist
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)

def load_and_inspect(csv_path):
    """Loads a CSV and prints basic info."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path}: {df.shape}")
    return df

# Load Data
df_train = load_and_inspect(train_csv)
df_test = load_and_inspect(test_csv)

def find_best_shape(num_features):
    """Finds the best (height, width) to reshape data without losing information."""
    for h in range(int(np.sqrt(num_features)), 0, -1):
        if num_features % h == 0:
            return h, num_features // h
    return num_features, 1  # If prime, keep it as (num_features, 1)

def normalize_and_convert_to_images(df, output_dir, prefix="train"):
    """Normalizes CSV data and converts rows to grayscale images."""
    scaler = MinMaxScaler(feature_range=(0, 255))
    data = scaler.fit_transform(df.iloc[:, 1:].values)  # Skip ID column if present

    IMG_SIZE = (256, 256)

    height, width = find_best_shape(data.shape[1])
    # height, width = IMG_SIZE
    print(f"Reshaping data into images of size: {height}x{width}")

    for idx, row in tqdm(enumerate(data), total=len(data), desc=f"Processing {prefix} images"):
        image = row.reshape((height, width))
        plt.imsave(os.path.join(output_dir, f"{prefix}_{idx}.png"), image, cmap='gray')

# Convert Train and Test Data
normalize_and_convert_to_images(df_train, output_train_dir, prefix="train")
normalize_and_convert_to_images(df_test, output_test_dir, prefix="test")

print("Conversion Completed! Images saved in respective folders.")