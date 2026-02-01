import streamlit as st
import cv2
import numpy as np
from PIL import Image
from drowsiness_utils import analyze_drowsiness

st.set_page_config(page_title="Driver Drowsiness Detection")

st.title("🚗 Driver Drowsiness Detection System")

uploaded_file = st.file_uploader("Upload Driver Image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Driver Image")

    img_np = np.array(image)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    status = analyze_drowsiness(img_np)

    if status == "DROWSY":
        st.error("⚠️ DRIVER IS DROWSY!")
    elif status == "AWAKE":
        st.success("✅ DRIVER IS AWAKE")
    else:
        st.warning("Face not detected")
