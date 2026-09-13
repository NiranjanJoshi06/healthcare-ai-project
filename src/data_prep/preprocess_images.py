"""
Day 5 — Preprocessing for the retina image dataset
Run with: python3 src/data_prep/preprocess_images.py
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_ROOT = "data/raw/retina_images/gaussian_filtered_images/gaussian_filtered_images"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

CLASSES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}

# ---- 1. Gather all image paths + labels ----
records = []
for cls in CLASSES:
    cls_dir = os.path.join(IMAGE_ROOT, cls)
    for fname in os.listdir(cls_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            records.append({
                "filepath": os.path.join(cls_dir, fname),
                "class_name": cls,
                "label": CLASS_TO_IDX[cls],
            })

df = pd.DataFrame(records)
print(f"Total images gathered: {len(df)}")
print(df["class_name"].value_counts())

# ---- 2. Stratified split: 70% train / 15% val / 15% test ----
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
)

print(f"\nSplit sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"\n{name} class distribution:")
    print(split_df["class_name"].value_counts())

# ---- 3. Save split CSVs (filepath + label, no physical copying needed) ----
train_df.to_csv(f"{OUT_DIR}/image_train.csv", index=False)
val_df.to_csv(f"{OUT_DIR}/image_val.csv", index=False)
test_df.to_csv(f"{OUT_DIR}/image_test.csv", index=False)

print(f"\nSaved: {OUT_DIR}/image_train.csv, image_val.csv, image_test.csv")

# ---- 4. Save class mapping for later use (dashboard, training script) ----
import json
with open(f"{OUT_DIR}/image_class_mapping.json", "w") as f:
    json.dump(CLASS_TO_IDX, f, indent=2)
print(f"Saved class mapping: {OUT_DIR}/image_class_mapping.json")

print("\nIMAGE PREPROCESSING COMPLETE.")
print("Note: severe class imbalance remains (Severe=5.3%, No_DR=49%).")
print("We'll handle this at training time (Day 8) using class weights and/or")
print("targeted data augmentation for minority classes, NOT by duplicating")
print("or discarding images now — that would distort the val/test evaluation.")
