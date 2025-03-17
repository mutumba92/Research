import os
import matplotlib.pyplot as plt
import seaborn as sns



DATASET_PATH = '/Users/mac/Desktop/datasets/malimg_dataset'
TRAIN_PATH = os.path.join(DATASET_PATH, 'train')
TEST_PATH = os.path.join(DATASET_PATH, 'test')
VALIDATE_PATH = os.path.join(DATASET_PATH, 'val')

# Function to count images per class in a directory
def count_images_in_directory(directory_path):
    class_counts = {}
    for class_name in os.listdir(directory_path):
        class_folder = os.path.join(directory_path, class_name)
        if os.path.isdir(class_folder):
            class_counts[class_name] = len(os.listdir(class_folder))
    return class_counts

# Get class distributions
train_counts = count_images_in_directory(TRAIN_PATH)
test_counts = count_images_in_directory(TEST_PATH)
validate_counts = count_images_in_directory(VALIDATE_PATH)

# Plotting the class distributions
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

sns.barplot(x=list(train_counts.keys()), y=list(train_counts.values()), ax=axes[0])
axes[0].set_title('Training Set Class Distribution')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)

sns.barplot(x=list(test_counts.keys()), y=list(test_counts.values()), ax=axes[1])
axes[1].set_title('Test Set Class Distribution')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90)

sns.barplot(x=list(validate_counts.keys()), y=list(validate_counts.values()), ax=axes[2])
axes[2].set_title('Validation Set Class Distribution')
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=90)

plt.tight_layout()
plt.show()
