"""
Day 13 — Explainable Healthcare AI Dashboard
Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import io

st.set_page_config(page_title="Explainable Healthcare AI", layout="wide")

MODEL_DIR = "models_saved"
DATA_DIR = "data/processed"
IMG_SIZE = 224

DOMAINS = {
    "Health": ["HighBP", "HighChol", "CholCheck", "BMI", "Stroke", "HeartDiseaseorAttack", "GenHlth"],
    "Lifestyle": ["Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump"],
    "Demographic": ["Sex", "Age", "Education", "Income"],
    "HealthcareAccess": ["AnyHealthcare", "NoDocbcCost"],
    "Other": ["MentHlth", "PhysHlth", "DiffWalk"],
}

# ---- Codebook mappings for human-readable inputs ----
AGE_OPTIONS = {
    "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5, "45-49": 6,
    "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10, "70-74": 11, "75-79": 12, "80+": 13,
}
EDUCATION_OPTIONS = {
    "Never attended / kindergarten only": 1, "Elementary": 2, "Some high school": 3,
    "High school graduate": 4, "Some college / technical school": 5, "College graduate": 6,
}
INCOME_OPTIONS = {
    "< $10,000": 1, "$10,000-14,999": 2, "$15,000-19,999": 3, "$20,000-24,999": 4,
    "$25,000-34,999": 5, "$35,000-49,999": 6, "$50,000-74,999": 7, "$75,000+": 8,
}
GENHLTH_OPTIONS = {"Excellent": 1, "Very good": 2, "Good": 3, "Fair": 4, "Poor": 5}
YES_NO = {"No": 0, "Yes": 1}

# ---- Cached resource loaders ----
@st.cache_resource
def load_tabular_resources():
    model = joblib.load(f"{MODEL_DIR}/combined_LogisticRegression.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/tabular_scaler.pkl")
    with open(f"{DATA_DIR}/feature_sets.json") as f:
        feature_sets = json.load(f)
    background = pd.read_csv(f"{DATA_DIR}/train.csv")[feature_sets["combined"]].sample(
        n=500, random_state=42
    )
    explainer = shap.LinearExplainer(model, background)
    return model, scaler, feature_sets["combined"], explainer

@st.cache_resource
def load_cnn_resources():
    model = tf.keras.models.load_model(f"{MODEL_DIR}/cnn_model.keras")
    with open(f"{DATA_DIR}/image_class_mapping.json") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return model, idx_to_class

CONTINUOUS_COLS = ["BMI", "MentHlth", "PhysHlth"]

def make_gradcam(model, img_pre):
    base_model = model.get_layer("mobilenetv2_1.00_224")
    last_conv_layer_name = None
    for layer in reversed(base_model.layers):
        if "conv" in layer.name.lower():
            last_conv_layer_name = layer.name
            break
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output],
    )
    top_layers = model.layers[3:]
    with tf.GradientTape() as tape:
        conv_output, base_output = grad_model(img_pre)
        tape.watch(conv_output)
        x = base_output
        for layer in top_layers:
            x = layer(x)
        preds = x
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index), preds.numpy()[0]

def overlay_heatmap(heatmap, img_display, alpha=0.4):
    heatmap_resized = tf.image.resize(
        heatmap[..., tf.newaxis], (IMG_SIZE, IMG_SIZE)
    ).numpy().squeeze()
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    jet_heatmap = np.uint8(jet_heatmap * 255)
    overlaid = jet_heatmap * alpha + img_display * (1 - alpha)
    return overlaid.astype("uint8")

# ==================== UI ====================

st.title("Explainable Multimodal Healthcare AI")
st.caption(
    "Research prototype — NOT a clinical diagnostic tool. "
    "Predictions and explanations reflect model behavior on public benchmark "
    "data, not verified medical advice."
)

page = st.sidebar.radio(
    "Select Assessment",
    ["Health & Lifestyle Risk (Diabetes)", "Retina Image Analysis (Diabetic Retinopathy)"],
)

# -------------------- PAGE 1: TABULAR --------------------
if page == "Health & Lifestyle Risk (Diabetes)":
    st.header("Health & Lifestyle Diabetes Risk Assessment")
    model, scaler, features, explainer = load_tabular_resources()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Health")
        high_bp = st.selectbox("High Blood Pressure?", list(YES_NO.keys()))
        high_chol = st.selectbox("High Cholesterol?", list(YES_NO.keys()))
        chol_check = st.selectbox("Cholesterol checked in last 5 years?", list(YES_NO.keys()), index=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=27.0, step=0.1)
        stroke = st.selectbox("Ever had a stroke?", list(YES_NO.keys()))
        heart_disease = st.selectbox("Heart disease or heart attack history?", list(YES_NO.keys()))
        genhlth = st.selectbox("General health (self-rated)", list(GENHLTH_OPTIONS.keys()), index=2)

    with col2:
        st.subheader("Lifestyle")
        smoker = st.selectbox("Smoked 100+ cigarettes in lifetime?", list(YES_NO.keys()))
        phys_activity = st.selectbox("Physical activity in last 30 days?", list(YES_NO.keys()), index=1)
        fruits = st.selectbox("Eats fruit 1+ times/day?", list(YES_NO.keys()), index=1)
        veggies = st.selectbox("Eats vegetables 1+ times/day?", list(YES_NO.keys()), index=1)
        hvy_alcohol = st.selectbox("Heavy alcohol consumption?", list(YES_NO.keys()))
        diff_walk = st.selectbox("Serious difficulty walking/climbing stairs?", list(YES_NO.keys()))

    with col3:
        st.subheader("Demographic & Access")
        sex = st.selectbox("Sex", ["Female", "Male"])
        age_range = st.selectbox("Age range", list(AGE_OPTIONS.keys()), index=7)
        education = st.selectbox("Education level", list(EDUCATION_OPTIONS.keys()), index=4)
        income = st.selectbox("Income bracket", list(INCOME_OPTIONS.keys()), index=5)
        any_healthcare = st.selectbox("Has healthcare coverage?", list(YES_NO.keys()), index=1)
        no_doc_cost = st.selectbox("Avoided doctor due to cost (last year)?", list(YES_NO.keys()))
        ment_hlth = st.slider("Days of poor mental health (last 30 days)", 0, 30, 0)
        phys_hlth = st.slider("Days of poor physical health (last 30 days)", 0, 30, 0)

    if st.button("Assess Risk", type="primary"):
        row = {
            "HighBP": YES_NO[high_bp], "HighChol": YES_NO[high_chol], "CholCheck": YES_NO[chol_check],
            "BMI": bmi, "Stroke": YES_NO[stroke], "HeartDiseaseorAttack": YES_NO[heart_disease],
            "GenHlth": GENHLTH_OPTIONS[genhlth], "Smoker": YES_NO[smoker],
            "PhysActivity": YES_NO[phys_activity], "Fruits": YES_NO[fruits], "Veggies": YES_NO[veggies],
            "HvyAlcoholConsump": YES_NO[hvy_alcohol], "Sex": 1 if sex == "Male" else 0,
            "Age": AGE_OPTIONS[age_range], "Education": EDUCATION_OPTIONS[education],
            "Income": INCOME_OPTIONS[income], "AnyHealthcare": YES_NO[any_healthcare],
            "NoDocbcCost": YES_NO[no_doc_cost], "MentHlth": ment_hlth, "PhysHlth": phys_hlth,
            "DiffWalk": YES_NO[diff_walk],
        }
        X = pd.DataFrame([row])[features]
        X_scaled = X.copy()
        X_scaled[CONTINUOUS_COLS] = scaler.transform(X[CONTINUOUS_COLS])

        proba = model.predict_proba(X_scaled)[0, 1]

        st.divider()
        st.subheader("Result")
        risk_col, gauge_col = st.columns([1, 2])
        with risk_col:
            st.metric("Estimated Diabetes Risk", f"{proba * 100:.1f}%")
            if proba >= 0.5:
                st.warning("Model flags this profile as higher-risk. This is not a diagnosis.")
            else:
                st.success("Model flags this profile as lower-risk. This is not a diagnosis.")

        # SHAP explanation for this single prediction
        shap_values = explainer.shap_values(X_scaled)[0]
        explain_df = pd.DataFrame({
            "feature": features,
            "value": X.values[0],
            "contribution": shap_values,
        }).sort_values("contribution", key=abs, ascending=False)

        with gauge_col:
            st.write("**Top factors driving this prediction:**")
            top10 = explain_df.head(10).sort_values("contribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ["#d62728" if v > 0 else "#2ca02c" for v in top10["contribution"]]
            ax.barh(top10["feature"], top10["contribution"], color=colors)
            ax.set_xlabel("SHAP contribution (red = increases risk, green = decreases)")
            st.pyplot(fig)

        # Domain-level breakdown for this prediction
        st.write("**Which type of information drove this result:**")
        domain_contrib = {}
        for domain, cols in DOMAINS.items():
            cols_present = [c for c in cols if c in features]
            domain_contrib[domain] = explain_df[
                explain_df["feature"].isin(cols_present)
            ]["contribution"].abs().sum()
        domain_series = pd.Series(domain_contrib).sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.bar(domain_series.index, domain_series.values, color="teal")
        ax2.set_ylabel("Total |contribution|")
        plt.xticks(rotation=20)
        st.pyplot(fig2)

        st.caption(
            "SHAP values explain what the MODEL relied on for this specific prediction, "
            "not biological causation. Not a substitute for medical advice."
        )

# -------------------- PAGE 2: IMAGE --------------------
else:
    st.header("Retina Image Analysis — Diabetic Retinopathy Severity")
    st.info(
        "Standalone module: not linked to patient-level tabular data. "
        "Trained on the APTOS 2019 benchmark dataset (3,662 images)."
    )
    cnn_model, idx_to_class = load_cnn_resources()

    uploaded_file = st.file_uploader("Upload a retina fundus image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img)
        img_pre = preprocess_input(tf.expand_dims(tf.constant(img_array, dtype=tf.float32), 0))

        heatmap, pred_idx, pred_probs = make_gradcam(cnn_model, img_pre)
        overlay = overlay_heatmap(heatmap, img_array)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_array, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.image(overlay, caption="Grad-CAM: Regions Influencing Prediction", use_container_width=True)

        st.subheader(f"Predicted severity: **{idx_to_class[pred_idx]}**")
        prob_df = pd.DataFrame({
            "class": [idx_to_class[i] for i in range(len(pred_probs))],
            "confidence": pred_probs,
        }).sort_values("confidence", ascending=False)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(prob_df["class"], prob_df["confidence"], color="steelblue")
        ax.set_xlabel("Confidence")
        st.pyplot(fig)

        st.caption(
            "Grad-CAM highlights pixels that influenced the model's decision — this is "
            "NOT a confirmed clinical finding. Model performs well on healthy-vs-diseased "
            "screening but has lower accuracy on rare severe classes (see project report)."
        )
    else:
        st.write("Upload an image to see the prediction and Grad-CAM explanation.")

st.sidebar.divider()
st.sidebar.caption(
    "Explainable Multimodal Healthcare AI — Research Prototype\n\n"
    "Tabular: BRFSS 2015 Diabetes Health Indicators\n\n"
    "Imaging: APTOS 2019 (standalone module)"
)
