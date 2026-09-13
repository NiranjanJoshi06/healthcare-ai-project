"""
Day 4 — Preprocessing for the tabular diabetes dataset
Run with: python3 src/data_prep/preprocess_tabular.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json

RAW_PATH = "data/raw/diabetes_tabular.csv"
OUT_DIR = "data/processed"
MODEL_DIR = "models_saved"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "Diabetes_binary"

# ---- Feature domains (matches EDA domain map) ----
DOMAINS = {
    "health": ["HighBP", "HighChol", "CholCheck", "BMI", "Stroke", "HeartDiseaseorAttack", "GenHlth"],
    "lifestyle": ["Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump"],
    "demographic": ["Sex", "Age", "Education", "Income"],
    "healthcare_access": ["AnyHealthcare", "NoDocbcCost"],
}
# Not assigned to a domain but still useful predictors — include in "combined" only
EXTRA = ["MentHlth", "PhysHlth", "DiffWalk"]

# Continuous columns that benefit from scaling (rest are binary/ordinal, left as-is)
CONTINUOUS_COLS = ["BMI", "MentHlth", "PhysHlth"]

# ---- 1. Load ----
df = pd.read_csv(RAW_PATH)
print("Original shape:", df.shape)

# ---- 2. Drop duplicates (24,206 found in EDA) ----
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Dropped {before - len(df)} duplicate rows. New shape: {df.shape}")

# ---- 3. Define experiment feature sets ----
health_features = DOMAINS["health"]
lifestyle_features = DOMAINS["lifestyle"]
combined_features = (
    DOMAINS["health"] + DOMAINS["lifestyle"] + DOMAINS["demographic"]
    + DOMAINS["healthcare_access"] + EXTRA
)

feature_sets = {
    "health_only": health_features,
    "lifestyle_only": lifestyle_features,
    "combined": combined_features,
}

with open(f"{OUT_DIR}/feature_sets.json", "w") as f:
    json.dump(feature_sets, f, indent=2)
print("\nFeature sets saved:")
for name, feats in feature_sets.items():
    print(f"  {name}: {len(feats)} features")

# ---- 4. Stratified split: 70% train / 15% val / 15% test ----
X = df[combined_features]  # superset; each experiment will subselect columns
y = df[TARGET]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print(f"\nSplit sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
print("Train class balance:\n", y_train.value_counts(normalize=True).round(3))
print("Val class balance:\n", y_val.value_counts(normalize=True).round(3))
print("Test class balance:\n", y_test.value_counts(normalize=True).round(3))

# ---- 5. Scale continuous columns (fit on train only, to avoid leakage) ----
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled[CONTINUOUS_COLS] = scaler.fit_transform(X_train[CONTINUOUS_COLS])
X_val_scaled[CONTINUOUS_COLS] = scaler.transform(X_val[CONTINUOUS_COLS])
X_test_scaled[CONTINUOUS_COLS] = scaler.transform(X_test[CONTINUOUS_COLS])

joblib.dump(scaler, f"{MODEL_DIR}/tabular_scaler.pkl")
print("\nScaler fitted on train only, saved to models_saved/tabular_scaler.pkl")

# ---- 6. Save processed splits (with target attached) ----
def save_split(X, y, name):
    out = X.copy()
    out[TARGET] = y.values
    out.to_csv(f"{OUT_DIR}/{name}.csv", index=False)
    print(f"Saved {OUT_DIR}/{name}.csv  shape={out.shape}")

save_split(X_train_scaled, y_train, "train")
save_split(X_val_scaled, y_val, "val")
save_split(X_test_scaled, y_test, "test")

print("\nPREPROCESSING COMPLETE.")
print("Note: class imbalance (86/14) was NOT fixed here via resampling.")
print("We'll handle it at model-training time using class_weight='balanced' (LR/RF)")
print("or scale_pos_weight (XGBoost), so val/test stay representative of real-world distribution.")
