"""
Day 8 — Train CNN on retina images (diabetic retinopathy severity, 5 classes)
Transfer learning with MobileNetV2 (frozen base) — fast enough for CPU training.
Run with: python3 src/models/train_cnn.py
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import json
import os

DATA_DIR = "data/processed"
MODEL_DIR = "models_saved"
FIG_DIR = "reports/figures"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15

with open(f"{DATA_DIR}/image_class_mapping.json") as f:
    CLASS_TO_IDX = json.load(f)
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASS_TO_IDX)

# ---- Load split CSVs ----
train_df = pd.read_csv(f"{DATA_DIR}/image_train.csv")
val_df = pd.read_csv(f"{DATA_DIR}/image_val.csv")
test_df = pd.read_csv(f"{DATA_DIR}/image_test.csv")

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ---- Build tf.data pipelines ----
def load_and_preprocess(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = preprocess_input(img)  # MobileNetV2-specific normalization
    return img, label

def make_dataset(df, training=False):
    ds = tf.data.Dataset.from_tensor_slices((df["filepath"].values, df["label"].values))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=42)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

# ---- Data augmentation (applied only during training, inside the model) ----
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# ---- Class weights (severe imbalance: Severe=5.3%, No_DR=49%) ----
class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_df["label"].values,
)
class_weights = dict(enumerate(class_weights_arr))
print("\nClass weights (to counter imbalance):")
for idx, w in class_weights.items():
    print(f"  {IDX_TO_CLASS[idx]:15s}: {w:.3f}")

# ---- Build model: MobileNetV2 base (frozen) + custom head ----
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # freeze — fast training, avoids overfitting on ~2.5k images

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---- Train ----
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=4, restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stop],
)

# ---- Save model ----
model.save(f"{MODEL_DIR}/cnn_model.keras")
print(f"\nSaved model: {MODEL_DIR}/cnn_model.keras")

# ---- Training curves ----
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/cnn_training_curves.png", bbox_inches="tight")
plt.close()
print(f"Saved training curves: {FIG_DIR}/cnn_training_curves.png")

# ---- Evaluate on test set ----
print("\n" + "=" * 60)
print("TEST SET EVALUATION")
print("=" * 60)

y_true = test_df["label"].values
y_pred_proba = model.predict(test_ds)
y_pred = np.argmax(y_pred_proba, axis=1)

class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks(range(NUM_CLASSES), class_names, rotation=45, ha="right")
plt.yticks(range(NUM_CLASSES), class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Retina Severity Classification")
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/cnn_confusion_matrix.png", bbox_inches="tight")
plt.close()
print(f"\nSaved confusion matrix: {FIG_DIR}/cnn_confusion_matrix.png")

print("\nCNN TRAINING COMPLETE.")
print("NOTE: This is a standalone imaging module (not patient-matched to the")
print("tabular data), evaluated on its own merits per our project plan.")
