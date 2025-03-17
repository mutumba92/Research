import numpy as np
import tensorflow as tf
from keras.applications import EfficientNetB4
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter

# Paths & Config
DATASET_PATH = "/Users/mac/Desktop/datasets/malimg_dataset"
IMG_SIZE = (224, 224)  # Increased image size for deeper models
BATCH_SIZE = 32
EPOCHS = 150  # Further increased epochs

# ----- 1. IMPROVED DATA AUGMENTATION -----
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=60,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.3,
    zoom_range=0.5,
    brightness_range=[0.4, 1.6],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Load Dataset with Correct Directories
train_generator = datagen.flow_from_directory(
    f"{DATASET_PATH}/train",
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    f"{DATASET_PATH}/val",
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    shuffle=True
)

test_generator = datagen.flow_from_directory(
    f"{DATASET_PATH}/test",
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    shuffle=False
)

# ----- 2. CLASS WEIGHTS -----
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# ----- 3. ENHANCED MODEL -----
def create_model():
    input_layer = layers.Input(shape=(224, 224, 1))
    x = layers.Conv2D(3, (1, 1), activation='relu')(input_layer)

    base_model = EfficientNetB4(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = True
    for layer in base_model.layers[:50]:  # Keeping more layers trainable
        layer.trainable = False

    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(0.6)(x)  # Increased dropout
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(train_generator.num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_model()

# ----- 4. CALLBACKS -----
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# ----- 5. MODEL TRAINING -----
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, lr_scheduler]
)

# ----- 6. EVALUATION -----
y_pred = np.argmax(model.predict(test_generator), axis=-1)
print(classification_report(test_generator.classes, y_pred))
cm = confusion_matrix(test_generator.classes, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
