import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

# Configuration
SPLITS_DIR = r"C:\Z_Soutenance\DATA\SPLITS"
MODELS_DIR = r"C:\Z_Soutenance\MODELS"
os.makedirs(MODELS_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 20
IMG_SIZE = (224, 224)

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(SPLITS_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_test_datagen.flow_from_directory(
    os.path.join(SPLITS_DIR, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(SPLITS_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Class weights (calculé dynamiquement)
num_normal = 22463  # De ton EDA/fusion
num_pneumonia = 20808
total = num_normal + num_pneumonia
class_weight = {0: total / (2 * num_normal), 1: total / (2 * num_pneumonia)}
print(f"Class weights : {class_weight}")

# Modèle base
def create_model(base_model, name):
    base = base_model(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = False  # Freeze

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(base.input, outputs, name=name)
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

models = {
    'ResNet50': create_model(ResNet50, 'ResNet50'),
    'MobileNetV2': create_model(MobileNetV2, 'MobileNetV2')
}

# Entraînement
for name, model in models.items():
    print(f"\n=== ENTRAÎNEMENT {name} ===")

    # Callbacks spécifiques au modèle (fix du bug filepath)
    callbacks = [
        EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True, mode='max'),
        ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, mode='max'),
        ModelCheckpoint(os.path.join(MODELS_DIR, f'{name}_best.h5'), monitor='val_auc', save_best_only=True, mode='max')
    ]

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    # Évaluation test
    test_loss, test_acc, test_auc = model.evaluate(test_generator, verbose=0)
    y_pred_prob = model.predict(test_generator)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = test_generator.classes
    print(f"\n{name} - Test Accuracy: {test_acc:.4f} | AUC: {test_auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))

    # Courbes
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{name} - Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Val AUC')
    plt.title(f'{name} - AUC')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, f'{name}_curves.png'), dpi=150)
    plt.show()

print("\nEntraînement terminé ! Modèles + courbes dans :", MODELS_DIR)