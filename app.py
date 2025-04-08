import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import time
from threading import Thread
import queue

# 🚦 Page Config
st.set_page_config(page_title="Traffic Sign Alert", layout="centered")

# 📌 Class Labels (same as your original)
CLASS_LABELS = {
    0: "Speed limit 20 km per hour",
    # ... (keep all your existing labels)
    42: "End of no passing by vehicles over 3.5 tons"
}

# 📦 Model Loading
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('best_model.h5')

model = load_model()

# 🖼️ Image Processing
def preprocess_image(image):
    image = np.array(image.convert('RGB'))
    image = cv2.resize(image, (64, 64))
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=0)

# 🗣️ Browser-Based Voice Alert
def voice_alert(text):
    js = f"""
    <script>
    if ('speechSynthesis' in window) {{
        var msg = new SpeechSynthesisUtterance("Warning: {text}");
        window.speechSynthesis.speak(msg);
    }}
    </script>
    """
    st.components.v1.html(js, height=0)

# 📹 Webcam Capture (for browsers that support it)
def get_webcam_frame():
    img_file_buffer = st.camera_input("Take a picture of traffic sign")
    if img_file_buffer is not None:
        return Image.open(img_file_buffer)
    return None

# 🎚️ Confidence Threshold
CONFIDENCE_THRESHOLD = 0.85

# 🖥️ Main UI
def main():
    st.title("🚦 Traffic Sign Recognition")
    st.write("Upload an image or use your camera to detect traffic signs")
    
    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Use Camera"])
    
    with tab1:
        uploaded_file = st.file_uploader("Choose traffic sign image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process image
            processed = preprocess_image(image)
            prediction = model.predict(processed)
            class_idx = np.argmax(prediction)
            confidence = prediction[0][class_idx]
            
            if confidence > CONFIDENCE_THRESHOLD:
                sign_name = CLASS_LABELS.get(class_idx, "Unknown sign")
                st.success(f"Detected: {sign_name} (Confidence: {confidence:.2%})")
                voice_alert(sign_name)
            else:
                st.warning(f"Uncertain detection (Confidence: {confidence:.2%})")
    
    with tab2:
        st.info("Note: Camera access requires permission in your browser")
        image = get_webcam_frame()
        if image:
            # Process camera image
            processed = preprocess_image(image)
            prediction = model.predict(processed)
            class_idx = np.argmax(prediction)
            confidence = prediction[0][class_idx]
            
            if confidence > CONFIDENCE_THRESHOLD:
                sign_name = CLASS_LABELS.get(class_idx, "Unknown sign")
                st.success(f"Detected: {sign_name} (Confidence: {confidence:.2%})")
                voice_alert(sign_name)
            else:
                st.warning(f"Uncertain detection (Confidence: {confidence:.2%})")

if __name__ == "__main__":
    main()
