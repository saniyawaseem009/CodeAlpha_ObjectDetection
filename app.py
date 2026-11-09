import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os

# Page config (Layout wide karo)
st.set_page_config(page_title="Object Detection & Tracking", page_icon="🎯", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #121212 100%);
    }
    /* Header styling */
    .header-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        margin-bottom: 30px;
    }
    .header-container h1 { margin: 0; font-size: 3rem; font-weight: 700; }
    .header-container p { margin: 10px 0 0 0; font-size: 1.2rem; opacity: 0.9; }
    
    /* Sidebar styling */
    .css-1d391kg { background-color: #1a1a2e; }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102,126,234,0.4);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <h1>🎯 Object Detection & Tracking</h1>
    <p>Powered by YOLOv8 & ByteTrack | Upload Image or Video</p>
</div>
""", unsafe_allow_html=True)

# Model load karo
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Sidebar
st.sidebar.markdown("### ⚙️ Settings")
mode = st.sidebar.radio("Select Input Mode:", ["📷 Upload Image", "🎥 Upload Video"])

if "Image" in mode:
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📸 Original Image")
            st.image(image, channels="BGR", use_container_width=True)
            
        if st.button("🚀 Detect Objects"):
            with st.spinner("Detecting objects..."):
                results = model.predict(image)
                annotated_image = results[0].plot()
                
                with col2:
                    st.markdown("### 🎯 Detected Objects")
                    st.image(annotated_image, channels="BGR", use_container_width=True)
                st.success("✅ Detection Complete!")

elif "Video" in mode:
    uploaded_file = st.sidebar.file_uploader("Choose a video", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🎥 Original Video")
            st.video(tfile.name)
            
        if st.button("🚀 Start Tracking"):
            with st.spinner("Processing video..."):
                cap = cv2.VideoCapture(tfile.name)
                st_frame = st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
                    annotated_frame = results[0].plot()
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    with col2:
                        st.markdown("### 🎯 Tracked Objects")
                        st_frame.image(annotated_frame, channels="RGB", use_container_width=True)
                
                cap.release()
                os.unlink(tfile.name)
                st.success("✅ Tracking Complete!")