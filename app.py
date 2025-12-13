import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Model

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
MODEL_PATH = "brain_tumor_detection_optimized.h5"
IMAGE_SIZE = (256, 256)

TUMOR_TYPES = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
TUMOR_COLORS = {
    'Glioma': "#FF6B6B",
    'Meningioma': "#4ECDC4",
    'Pituitary': "#45B7D1",
    'No Tumor': "#95E1D3"
}

# ---------------------------------------------
# LOAD MODEL
# ---------------------------------------------
@st.cache_resource
def load_model():
    model = keras.models.load_model(MODEL_PATH)
    attention_layer = model.get_layer("branch_attention")
    attention_model = Model(inputs=model.input, outputs=attention_layer.output)
    return model, attention_model

model, attention_model = load_model()


# ---------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------
def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    original = img.copy()

    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype("float32") / 255.0
    img_rgb = np.stack([img]*3, axis=-1)
    img_batch = np.expand_dims(img_rgb, axis=0)
    
    return img_batch, original


# ---------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------
def predict(img_batch):
    probs = model.predict(img_batch)[0]
    attn = attention_model.predict(img_batch)[0]

    pred_idx = np.argmax(probs)
    tumor_type = TUMOR_TYPES[pred_idx]
    confidence = probs[pred_idx]

    return tumor_type, confidence, probs, attn


# ---------------------------------------------
# STREAMLIT UI
# ---------------------------------------------
st.title("🧠 Brain Tumor Detection - MRI Classifier")
st.write("Upload a Brain MRI image to detect the tumor type using a 3-model ensemble with Attention Fusion.")

uploaded_file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display original image
    st.image(uploaded_file, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess
    img_batch, original = preprocess_image(uploaded_file)

    # Predict
    tumor_type, confidence, probs, attn = predict(img_batch)

    # -------------------------------------
    # Display Results
    # -------------------------------------
    st.subheader("🎯 Prediction Results")
    st.write(f"**Tumor Type:** {tumor_type}")
    st.write(f"**Confidence:** {confidence:.2%}")

    # Probability Chart
    st.subheader("📊 Probability Distribution")
    prob_fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(TUMOR_TYPES, probs, color=[TUMOR_COLORS[t] for t in TUMOR_TYPES])
    ax.set_xlim(0, 1)
    for bar, p in zip(bars, probs):
        ax.text(p + 0.02, bar.get_y() + bar.get_height()/2, f"{p:.2%}", va='center')
    st.pyplot(prob_fig)

    # Attention Weights
    st.subheader("🔍 Attention Weights (Importance of Each Branch)")
    att_fig, ax = plt.subplots(figsize=(6, 4))
    branch_names = ["YOLO", "Mask R-CNN", "Vision Transformer"]
    att_bars = ax.bar(branch_names, attn, color=["#FF6B6B", "#4ECDC4", "#45B7D1"])
    for bar, w in zip(att_bars, attn):
        ax.text(bar.get_x() + bar.get_width()/2, w + 0.01, f"{w:.3f}", ha='center')
    st.pyplot(att_fig)

    st.success("Prediction Completed Successfully!")

else:
    st.info("Please upload a brain MRI image to begin.")
