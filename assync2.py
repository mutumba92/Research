import numpy as np
import tensorflow as tf
from keras.applications import MobileNetV2, EfficientNetB4, ResNet50
from keras import layers, models, optimizers, regularizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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
    fill_mode='nearest'
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

# Class Weights for Imbalanced Data
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
    x = base_model(x, training=False)
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

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.3, min_lr=1e-6)

# Custom Callback to Unfreeze Layers
class UnfreezeLayersCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, unfreeze_epoch=5):
        super().__init__()
        self.model = model
        self.unfreeze_epoch = unfreeze_epoch
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.unfreeze_epoch - 1:  # Unfreeze after set epoch
            print("Unfreezing more layers...")
            self.model.layers[2].trainable = True  # Unfreeze base model
            self.model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])

# Performance Metrics
def evaluate_model(model):
    y_true = test_generator.classes
    y_pred = np.argmax(model.predict(test_generator), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices, yticklabels=test_generator.class_indices)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Federated Learning with Async Weighting
class FedAsyncCustom(FedAvg):
    def __init__(self, initial_parameters=None):
        super().__init__(min_available_clients=2, fraction_fit=0.2, fraction_evaluate=0.2)
        self.global_weights = initial_parameters
    
    def aggregate_fit(self, rnd, results, failures):
        print(f"\n[Round {rnd}] Aggregating {len(results)} updates...")
        if not results:
            print("[ERROR] No results received from clients!")
            return self.global_weights, {}
        
        updates = [parameters_to_ndarrays(res.parameters) for _, res in results if res.parameters is not None]
        sample_sizes = [res.num_examples for _, res in results]
        total_samples = sum(sample_sizes)
        new_weights = [np.sum([update[i] * (sample_sizes[j] / total_samples) for j, update in enumerate(updates)], axis=0) for i in range(len(updates[0]))]
        
        self.global_weights = new_weights
        return ndarrays_to_parameters(self.global_weights), {}

strategy = FedAsyncCustom(initial_parameters=None)

# Client Side Code
class MalimgClient(fl.client.NumPyClient):
    def __init__(self, model_type="low"):
        self.model = create_model(model_type)
    
    def get_parameters(self, config):
        return self.model.get_weights()
    
    def set_parameters(self, parameters):
        self.model.set_weights(parameters)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator, class_weight=class_weight_dict, callbacks=[early_stopping, reduce_lr])
        return self.get_parameters(config), len(train_generator), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(test_generator)
        return float(loss), len(test_generator), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="0.0.0.0:9090", client=MalimgClient(model_type="low"))
