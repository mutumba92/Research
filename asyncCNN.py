import numpy as np
import tensorflow as tf
from keras.applications import MobileNetV2, EfficientNetB4, ResNet50
from keras import layers, models, optimizers, regularizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
import matplotlib.pyplot as plt
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays


# Paths & Config
DATASET_PATH = "/Users/mac/Desktop/datasets/malimg_dataset"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

# Data Augmentation with Feature Standardization
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest',
    featurewise_center=True,
    featurewise_std_normalization=True,
)

# Load Dataset
train_generator = datagen.flow_from_directory(
    f"{DATASET_PATH}/train", target_size=IMG_SIZE, color_mode="grayscale", batch_size=BATCH_SIZE, class_mode="sparse", shuffle=True
)
val_generator = datagen.flow_from_directory(
    f"{DATASET_PATH}/val", target_size=IMG_SIZE, color_mode="grayscale", batch_size=BATCH_SIZE, class_mode="sparse", shuffle=True
)
test_generator = datagen.flow_from_directory(
    f"{DATASET_PATH}/test", target_size=IMG_SIZE, color_mode="grayscale", batch_size=BATCH_SIZE, class_mode="sparse", shuffle=False
)

# Class Weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Model Creation with Regularization
def create_model(model_type):
    input_layer = layers.Input(shape=(128, 128, 1))
    x = layers.Conv2D(3, (1, 1), activation='relu')(input_layer)
    
    if model_type == "low":
        base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
    elif model_type == "mid":
        base_model = EfficientNetB4(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
    else:
        base_model = ResNet50(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
    
    base_model.trainable = False  # Initially freeze the base model
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(train_generator.num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Callbacks with Optimized Settings
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.3, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Custom Callback to Unfreeze Layers After 5 Epochs
class UnfreezeLayersCallback(Callback):
    def __init__(self, model, unfreeze_epoch=5):
        super().__init__()
        self.model = model
        self.unfreeze_epoch = unfreeze_epoch
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.unfreeze_epoch - 1:  # After the specified epoch
            print("Unfreezing more layers...")
            # Unfreeze layers progressively
            for layer in self.model.layers:
                if isinstance(layer, layers.Conv2D):
                    layer.trainable = True  # Unfreeze Conv2D layers
            self.model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])
            print("Unfreezing done.")

# Federated Learning with Dynamic Async Weighting
class FedAsyncCustom(FedAvg):
    def __init__(self, initial_parameters=None):
        super().__init__(min_available_clients=2, fraction_fit=0.2, fraction_evaluate=0.2)
        self.global_weights = initial_parameters
    
    def aggregate_fit(self, rnd, results, failures):
        print(f"\n[Round {rnd}] Aggregating {len(results)} updates...")

        if not results:
            print("[ERROR] No results received from clients!")
            return self.global_weights, {}

        # Extract updates & sample sizes
        updates = []
        sample_sizes = []
        
        for _, res in results:
            if res.parameters is not None:
                updates.append(parameters_to_ndarrays(res.parameters))
                sample_sizes.append(res.num_examples)  # Get number of training samples per client

        if not updates:
            print("[ERROR] No valid weight updates found!")
            return self.global_weights, {}

        total_samples = sum(sample_sizes)

        # Weighted aggregation
        new_weights = []
        for layer_idx in range(len(updates[0])):  # Iterate over model layers
            layer_updates = [update[layer_idx] * (sample_sizes[i] / total_samples) for i, update in enumerate(updates)]
            avg_layer = np.sum(layer_updates, axis=0)  # Weighted sum instead of simple mean
            new_weights.append(avg_layer)

        self.global_weights = new_weights
        print("[INFO] Aggregated new global weights successfully.")

        return ndarrays_to_parameters(self.global_weights), {}

strategy = FedAsyncCustom(initial_parameters=None)

# Metrics Plotting
def plot_metrics(history):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.show()

# Client Side Code
class MalimgClient(fl.client.NumPyClient):
    def __init__(self, model_type="low"):
        self.model = create_model(model_type)
        self.unfreeze_callback = UnfreezeLayersCallback(self.model)
    
    def get_parameters(self, config):
        return self.model.get_weights()
    
    def set_parameters(self, parameters):
        self.model.set_weights(parameters)
    
    def fit(self, parameters, config):
        try:
            self.set_parameters(parameters)
            history = self.model.fit(
                train_generator,
                epochs=EPOCHS,
                validation_data=val_generator,
                batch_size=BATCH_SIZE,
                class_weight=class_weight_dict,
                callbacks=[lr_scheduler, early_stopping, self.unfreeze_callback],
            )
            plot_metrics(history)
            return self.get_parameters(config), len(train_generator), {}
        except Exception as e:
            print(f"Error during training: {e}")
            return self.get_parameters(config), len(train_generator), {}

    def evaluate(self, parameters, config):
        try:
            self.set_parameters(parameters)
            loss, accuracy = self.model.evaluate(test_generator, batch_size=BATCH_SIZE)
            # Ensure correct return format
            num_examples = len(test_generator) * BATCH_SIZE 
            return float(loss), num_examples, {"accuracy": float(accuracy)}
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return None, None, {}

# Start Federated Learning Client
def start_client(model_type="low"):
    client = MalimgClient(model_type)
    fl.client.start_numpy_client(server_address="0.0.0.0:9090", client=client)

if __name__ == "__main__":
    start_client(model_type="low")