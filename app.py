import streamlit as st
import numpy as np
import cv2
from tensorflow import keras
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")

st.markdown('<h1 style="text-align: center;">🧠 Brain Tumor Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #7f8c8d;">3-Branch Ensemble Model with Attention Fusion</p>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('brain_tumor_detection_optimized.h5')
        st.sidebar.success("✓ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Please upload your model file: brain_tumor_detection_optimized.h5")
        return None

IMAGE_SIZE = (256, 256)
TUMOR_TYPES = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
TUMOR_DESCRIPTIONS = {
    'Glioma': 'Malignant tumor from glial cells (brain support cells)',
    'Meningioma': 'Tumor from meninges (brain lining)',
    'Pituitary': 'Tumor from pituitary gland (hormone center)',
    'No Tumor': 'Healthy brain - no tumor detected'
}

with st.sidebar:
    st.markdown("### 📋 About")
    st.info("""
    **3-Branch Ensemble Model:**
    - 🎯 YOLO: Location detection
    - 🔍 Mask R-CNN: Shape analysis
    - 🤖 ViT: Global context
    """)

model = load_model()
if model is None:
    st.stop()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader("Upload brain MRI image", type=['jpg', 'jpeg', 'png', 'tif', 'bmp'], help="Select a brain MRI scan image")
    
    image_to_process = None
    image_name = None
    
    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.read()
            image_array = np.frombuffer(image_bytes, np.uint8)
            image_to_process = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
            image_name = uploaded_file.name
            st.success(f"✓ Uploaded: {uploaded_file.name}")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
    
    if image_to_process is not None:
        st.markdown("#### 🖼️ Preview")
        st.image(image_to_process, use_column_width=True, channels='GRAY')
        st.caption(f"Size: {image_to_process.shape}")

with col2:
    st.markdown("### 🔍 Prediction Results")
    
    if st.button("🧠 RUN PREDICTION", use_container_width=True, type="primary"):
        if image_to_process is None:
            st.error("❌ Please upload an image first!")
        else:
            with st.spinner("⏳ Processing through 3-branch ensemble..."):
                try:
                    img_proc = cv2.resize(image_to_process, IMAGE_SIZE)
                    img_proc = img_proc.astype('float32') / 255.0
                    img_proc = np.stack([img_proc] * 3, axis=-1)
                    img_batch = np.expand_dims(img_proc, axis=0)
                    
                    predictions = model.predict(img_batch, verbose=0)[0]
                    tumor_idx = np.argmax(predictions)
                    tumor_type = TUMOR_TYPES[tumor_idx]
                    confidence = predictions[tumor_idx]
                    
                    st.success("✓ Prediction complete!")
                    
                    st.markdown(f"""
                        <div style='background-color: #e8f4f8; padding: 20px; border-radius: 10px; border-left: 5px solid #45B7D1;'>
                        <h2 style='margin: 0; color: #2c3e50;'>🎯 {tumor_type}</h2>
                        <h3 style='margin: 10px 0 0 0; color: #45B7D1;'>{confidence:.1%} Confident</h3>
                        <p style='margin: 10px 0 0 0; color: #555;'>{TUMOR_DESCRIPTIONS[tumor_type]}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if confidence >= 0.9:
                        st.success("✓✓✓ Very Confident")
                    elif confidence >= 0.8:
                        st.success("✓✓ Confident")
                    elif confidence >= 0.7:
                        st.info("✓ Moderately Confident")
                    else:
                        st.warning("⚠ Low Confidence - Review Recommended")
                    
                    st.markdown("**Confidence Score:**")
                    st.progress(float(confidence))
                    
                    st.markdown("**Probability for Each Class:**")
                    
                    prob_cols = st.columns(4)
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3']
                    
                    for col, tumor, prob, color in zip(prob_cols, TUMOR_TYPES, predictions, colors):
                        with col:
                            st.metric(tumor, f"{prob:.1%}", delta=None)
                    
                    st.markdown("**Probability Distribution:**")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    bars = ax.barh(TUMOR_TYPES, predictions, color=colors, edgecolor='black', linewidth=1.5)
                    ax.set_xlim(0, 1)
                    ax.set_xlabel('Probability', fontweight='bold')
                    
                    for bar, pred in zip(bars, predictions):
                        width = bar.get_width()
                        ax.text(width, bar.get_y() + bar.get_height()/2,
                               f' {pred:.1%}', ha='left', va='center', fontweight='bold')
                    
                    ax.grid(axis='x', alpha=0.3)
                    st.pyplot(fig, use_container_width=True)
                    
                    result_text = f'''BRAIN TUMOR DETECTION RESULT
============================

Image: {image_name}

PREDICTION:
-----------
Tumor Type: {tumor_type}
Confidence: {confidence:.1%}
Description: {TUMOR_DESCRIPTIONS[tumor_type]}

PROBABILITIES:
--------------'''
                    
                    for tumor, prob in zip(TUMOR_TYPES, predictions):
                        result_text += f"\n{tumor}: {prob:.1%}"
                    
                    st.download_button(
                        "📥 Download Results",
                        result_text,
                        f"tumor_result_{tumor_type.lower()}.txt",
                        "text/plain",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
                    st.info("Try uploading a different image or reloading the page")

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7f8c8d; margin-top: 30px;'>
    <small>⚠️ This is an AI-assisted tool, NOT a medical diagnosis</small><br>
    <small>Always consult with medical professionals for diagnosis</small><br>
    <small>Model: 3-Branch Ensemble (YOLO + Mask R-CNN + ViT)</small>
    </div>
    """, unsafe_allow_html=True)
