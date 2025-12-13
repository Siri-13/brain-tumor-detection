import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
import matplotlib.pyplot as plt
from PIL import Image

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
MODEL_PATH = "model.onnx"
IMAGE_SIZE = (256, 256)

TUMOR_TYPES = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3']

# -------------------------------------------------------
# LOAD ONNX MODEL
# -------------------------------------------------------
@st.cache_resource
def load_model():
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name

session, input_name, output_name = load_model()

# -------------------------------------------------------
# PREPROCESS IMAGE
# -------------------------------------------------------
def preprocess(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype("float32") / 255.0
    img = np.stack([img]*3, axis=-1)   # RGB
    img = np.expand_dims(img, axis=0)  # (1,256,256,3)
    return img

# -------------------------------------------------------
# RUN PREDICTION
# -------------------------------------------------------
def predict(img):
    pred = session.run([output_name], {input_name: img})[0][0]
    return pred

# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------
st.title("🧠 Brain Tumor Detection - ONNX Model")
st.write("Upload an MRI image to detect the tumor type using the ONNX version of your ensemble model.")

uploaded_file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded MRI", use_column_width=True)

    img_pre = preprocess(img)

    # Prediction
    probs = predict(img_pre)
    pred_idx = np.argmax(probs)
    tumor_type = TUMOR_TYPES[pred_idx]
    confidence = probs[pred_idx]

    st.subheader("🎯 Prediction")
    st.success(f"Tumor Type: **{tumor_type}**")
    st.write(f"Confidence: **{confidence:.2%}**")

    # Plot probability chart
    fig, ax = plt.subplots(figsize=(6,4))
    bars = ax.barh(TUMOR_TYPES, probs, color=COLORS)
    for bar, p in zip(bars, probs):
        ax.text(p + 0.02, bar.get_y() + bar.get_height()/2, f"{p:.2%}", va='center')
    st.pyplot(fig)
