"""
Day 3 — EDA on the retina image dataset (diabetic retinopathy severity)
Run with: python3 notebooks/02_eda_images.py
"""

import os
from PIL import Image
import matplotlib.pyplot as plt
import random

IMAGE_ROOT = "data/raw/retina_images/gaussian_filtered_images/gaussian_filtered_images"
FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

CLASSES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]

# ---- Count images per class ----
print("=" * 60)
print("IMAGE COUNT PER CLASS (severity of diabetic retinopathy)")
print("=" * 60)

counts = {}
sizes_sample = []
formats_sample = set()

for cls in CLASSES:
    cls_dir = os.path.join(IMAGE_ROOT, cls)
    if not os.path.isdir(cls_dir):
        print(f"WARNING: folder not found: {cls_dir}")
        continue
    files = [f for f in os.listdir(cls_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    counts[cls] = len(files)
    print(f"  {cls:15s}: {len(files)} images")

    # sample a few images to check size/format consistency
    for f in files[:5]:
        img_path = os.path.join(cls_dir, f)
        with Image.open(img_path) as img:
            sizes_sample.append(img.size)
            formats_sample.add(img.format)

total = sum(counts.values())
print(f"\nTOTAL IMAGES: {total}")

print("\nCLASS BALANCE (%):")
for cls, c in counts.items():
    print(f"  {cls:15s}: {100 * c / total:.1f}%")

# ---- Bar chart of class distribution ----
plt.figure(figsize=(7, 5))
plt.bar(counts.keys(), counts.values(), color="steelblue")
plt.title("Retina Image Class Distribution (DR Severity)")
plt.ylabel("Number of images")
plt.xticks(rotation=20)
plt.savefig(f"{FIG_DIR}/retina_class_distribution.png", bbox_inches="tight")
plt.close()

# ---- Image size / format consistency check ----
print("\nSAMPLE IMAGE SIZES (first 5 per class):", set(sizes_sample))
print("SAMPLE IMAGE FORMATS:", formats_sample)

# ---- Visual grid: one example per class ----
fig, axes = plt.subplots(1, len(CLASSES), figsize=(4 * len(CLASSES), 4))
for ax, cls in zip(axes, CLASSES):
    cls_dir = os.path.join(IMAGE_ROOT, cls)
    files = [f for f in os.listdir(cls_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if files:
        sample_file = random.choice(files)
        img = Image.open(os.path.join(cls_dir, sample_file))
        ax.imshow(img)
        ax.set_title(cls)
        ax.axis("off")
plt.suptitle("Sample Retina Image per Severity Class")
plt.savefig(f"{FIG_DIR}/retina_sample_grid.png", bbox_inches="tight")
plt.close()

print("\n" + "=" * 60)
print("IMAGE EDA COMPLETE. Figures saved to:", FIG_DIR)
print("  - retina_class_distribution.png")
print("  - retina_sample_grid.png")
print("=" * 60)
print("\nNOTE: This dataset is standalone (not patient-matched to the tabular")
print("BRFSS data), so it stays a separate module per our project plan.")
