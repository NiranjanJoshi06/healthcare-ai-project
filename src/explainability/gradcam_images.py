"""
Day 11 — Grad-CAM visualization on the trained CNN
Shows which regions of retina images drove each prediction.
Run with: python3 src/explainability/gradcam_images.py
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import os

DATA_DIR = "data/processed"
MODEL_DIR = "models_saved"
FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

IMG_SIZE = 224

with open(f"{DATA_DIR}/image_class_mapping.json") as f:
    CLASS_TO_IDX = json.load(f)
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# ---- Load trained model ----
model = tf.keras.models.load_model(f"{MODEL_DIR}/cnn_model.keras")

# ---- Find the last conv layer inside the MobileNetV2 base ----
base_model = model.get_layer("mobilenetv2_1.00_224")
last_conv_layer_name = None
for layer in reversed(base_model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower():
        last_conv_layer_name = layer.name
        break
print(f"Using last conv layer: {last_conv_layer_name}")

# Build a model that maps input -> (last conv activations, final predictions)
grad_model = tf.keras.models.Model(
    inputs=base_model.input,
    outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output],
)

def make_gradcam_heatmap(img_array, grad_model, top_layers, pred_index=None):
    """img_array: preprocessed image, shape (1, H, W, 3) BEFORE base_model.
    top_layers: the layers of `model` AFTER base_model (pooling, dropout, dense)."""
    with tf.GradientTape() as tape:
        conv_output, base_output = grad_model(img_array)
        tape.watch(conv_output)
        x = base_output
        for layer in top_layers:
            x = layer(x)
        preds = x
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index), preds.numpy()[0]

# Layers of `model` that come after the base_model (skip data_augmentation + base)
top_layers = model.layers[3:]  # [global_avg_pool, dropout, dense] — after input, augmentation, base

def load_image(filepath):
    img_raw = tf.io.read_file(filepath)
    img = tf.image.decode_png(img_raw, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img_display = img.numpy().astype("uint8")
    img_pre = preprocess_input(tf.expand_dims(img, 0))
    return img_pre, img_display

def overlay_heatmap(heatmap, img_display, alpha=0.4):
    heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], (IMG_SIZE, IMG_SIZE)).numpy().squeeze()
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    jet_heatmap = np.uint8(jet_heatmap * 255)
    overlaid = jet_heatmap * alpha + img_display * (1 - alpha)
    return overlaid.astype("uint8")

# ---- Pick one example per class from the test set ----
test_df = pd.read_csv(f"{DATA_DIR}/image_test.csv")

fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for col_idx, class_name in enumerate(IDX_TO_CLASS.values()):
    row = test_df[test_df["class_name"] == class_name].iloc[0]
    img_pre, img_display = load_image(row["filepath"])

    heatmap, pred_idx, pred_probs = make_gradcam_heatmap(img_pre, grad_model, top_layers)
    overlay = overlay_heatmap(heatmap, img_display)

    axes[0, col_idx].imshow(img_display)
    axes[0, col_idx].set_title(f"Actual: {class_name}")
    axes[0, col_idx].axis("off")

    axes[1, col_idx].imshow(overlay)
    axes[1, col_idx].set_title(f"Grad-CAM\nPredicted: {IDX_TO_CLASS[pred_idx]}")
    axes[1, col_idx].axis("off")

plt.suptitle("Grad-CAM: Which Image Regions Drove Each Prediction", fontsize=14)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/gradcam_examples.png", bbox_inches="tight", dpi=150)
plt.close()

print(f"\nSaved Grad-CAM visualization: {FIG_DIR}/gradcam_examples.png")
print("\nGRAD-CAM COMPLETE.")
print("IMPORTANT: Grad-CAM shows which pixels influenced the model's decision.")
print("A highlighted region is NOT automatically a clinically meaningful lesion —")
print("we present this as model behavior, not a diagnostic claim.")
