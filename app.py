import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
import matplotlib.pyplot as plt
from PIL import Image

# -------------------------------------------------------
# CONFIGURATION
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
# SAFE PREPROCESSING FUNCTION (NO cv2 errors)
# -------------------------------------------------------
def preprocess(img):
    # Convert PIL → NumPy
    img = np.array(img)

    # Convert to grayscale safely
    if len(img.shape) == 2:
        gray = img
    else:
        # Convert RGB → GRAY (correct conversion for Streamlit)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize to model input
    gray = cv2.resize(gray, IMAGE_SIZE)

    # Normalize
    gray = gray.astype("float32") / 255.0

    # Convert 1-channel image → 3 channel RGB
    gray = np.stack([gray] * 3, axis=-1)

    # Add batch dimension
    gray = np.expand_dims(gray, axis=0)

    return gray

# -------------------------------------------------------
# RUN MODEL PREDICTION
# -------------------------------------------------------
def predict(img):
    pred = session.run([output_name], {input_name: img})[0][0]
    return pred

# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------
st.title("🧠 Brain Tumor Detection Model")
st.write("Upload an MRI image to detect the tumor type using an optimized ensemble model.")

uploaded_file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded MRI", use_column_width=True)

    img_preprocessed = preprocess(img)
    probs = predict(img_preprocessed)

    pred_idx = np.argmax(probs)
    tumor_type = TUMOR_TYPES[pred_idx]
    confidence = probs[pred_idx]

    # Display prediction
    st.subheader("🎯 Prediction Result")
    st.success(f"Tumor Type: **{tumor_type}**")
    st.write(f"Confidence: **{confidence:.2%}**")

    # Probability graph
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(TUMOR_TYPES, probs, color=COLORS)

    for bar, p in zip(bars, probs):
        ax.text(p + 0.01, bar.get_y() + bar.get_height() / 2, f"{p:.2%}", va='center')

    ax.set_xlabel("Probability")
    ax.set_title("Prediction Confidence Levels")

    st.pyplot(fig)
