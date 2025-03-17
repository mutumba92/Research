import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import flwr as fl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for headless environments

# Paths & Config
DATASET_PATH = "/Users/mac/Desktop/datasets/malimg_dataset"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 5
NUM_CLIENTS = 3

# Enable Mixed Precision Training
from keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# ----- 1. DATA LOADING -----
datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = datagen.flow_from_directory(
    f"{DATASET_PATH}/train", target_size=IMG_SIZE, color_mode="grayscale",
    batch_size=BATCH_SIZE, class_mode="sparse", shuffle=True
)

val_generator = datagen.flow_from_directory(
    f"{DATASET_PATH}/val", target_size=IMG_SIZE, color_mode="grayscale",
    batch_size=BATCH_SIZE, class_mode="sparse", shuffle=False
)

# ----- 2. CLASS WEIGHTS -----
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(train_generator.classes),
                                     y=train_generator.classes)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# ----- 3. CNN MODEL -----
def create_model():
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(train_generator.num_classes, activation='softmax')
    ])
    model.compile(optimizer=optimizers.legacy.Adam(learning_rate=0.001),  # Use legacy Adam for M1/M2 Macs
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ----- 4. FLOWER CLIENT -----
class MalimgClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, val_data):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.history = {'accuracy': [], 'val_accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(self.train_data, epochs=EPOCHS, validation_data=self.val_data)
        
        # Store metrics
        self.history['accuracy'].append(history.history['accuracy'][-1])
        self.history['val_accuracy'].append(history.history['val_accuracy'][-1])
        
        return self.model.get_weights(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.val_data)

        # Get precision, recall, F1-score
        predictions = self.model.predict(self.val_data)
        y_true = self.val_data.classes
        y_pred = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

        # Store metrics
        self.history['precision'].append(precision)
        self.history['recall'].append(recall)
        self.history['f1'].append(f1)

        return loss, len(self.val_data), {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def plot_metrics(self):
        """Plot accuracy, precision, recall, and F1-score over training rounds."""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.history['accuracy'], label="Train Accuracy")
        plt.plot(self.history['val_accuracy'], label="Validation Accuracy")
        plt.xlabel('Rounds')
        plt.ylabel('Accuracy')
        plt.title('Federated Training Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history['precision'], label="Precision")
        plt.plot(self.history['recall'], label="Recall")
        plt.plot(self.history['f1'], label="F1-score")
        plt.xlabel('Rounds')
        plt.ylabel('Score')
        plt.title('Precision, Recall & F1-score')
        plt.legend()

        plt.tight_layout()
        plt.savefig("metrics_plot.png")  # Save instead of showing
        plt.close()

    def plot_confusion_matrix(self):
        """Plot confusion matrix after evaluation."""
        predictions = self.model.predict(self.val_data)
        y_true = self.val_data.classes
        y_pred = np.argmax(predictions, axis=1)
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
                    xticklabels=self.val_data.class_indices.keys(), 
                    yticklabels=self.val_data.class_indices.keys())
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_matrix.png")  # Save instead of showing
        plt.close()

# Start the Flower client
model = create_model()
client = MalimgClient(model, train_generator, val_generator)

# Start client training
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

# Plot metrics after federated training
client.plot_metrics()
client.plot_confusion_matrix()

print("Plots saved as 'metrics_plot.png' and 'confusion_matrix.png'.")