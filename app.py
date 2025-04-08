import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import time

# 🚦 Page Config
st.set_page_config(page_title="Traffic Sign Alert", layout="centered")

# 📌 Class Labels (reduced for example - include all your classes)
CLASS_LABELS = {
    0: "Speed limit 20 km/h",
    1: "Speed limit 30 km/h",
    2: "Speed limit 50 km/h",
    # ... include all your classes
}

# 📦 Model Loading with error handling
@st.cache_resource
def load_model():
    try:
        # Verify file exists
        if not os.path.exists('best_model.h5'):
            st.error("Model file not found!")
            return None
            
        # Load with explicit file handling
        with open('best_model.h5', 'rb') as f:
            model = tf.keras.models.load_model(f)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# 🖼️ Image Processing
def preprocess_image(image):
    try:
        image = np.array(image.convert('RGB'))
        image = tf.image.resize(image, [64, 64])  # Using TensorFlow resize
        image = image / 255.0
        return np.expand_dims(image, axis=0)
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None

# 🗣️ Voice Alert
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

# 🖥️ Main App
def main():
    st.title("🚦 Traffic Sign Recognition")
    
    if model is None:
        st.error("Failed to load model. Cannot proceed.")
        return
        
    upload_option = st.radio("Select input method:", 
                           ("Upload Image", "Take Photo"))

    if upload_option == "Upload Image":
        uploaded_file = st.file_uploader("Choose traffic sign image", 
                                       type=["jpg", "jpeg", "png"])
        if uploaded_file:
            process_image(Image.open(uploaded_file))
    else:
        img_file = st.camera_input("Take a photo of traffic sign")
        if img_file:
            process_image(Image.open(img_file))

def process_image(image):
    st.image(image, caption="Input Image", use_column_width=True)
    
    processed = preprocess_image(image)
    if processed is None:
        return
        
    with st.spinner("Analyzing..."):
        try:
            prediction = model.predict(processed)
            class_idx = np.argmax(prediction)
            confidence = prediction[0][class_idx]
            
            if confidence > 0.8:  # 80% confidence threshold
                sign_name = CLASS_LABELS.get(class_idx, "Unknown sign")
                st.success(f"Detected: {sign_name} ({confidence:.1%} confidence)")
                voice_alert(sign_name)
            else:
                st.warning(f"Low confidence detection ({confidence:.1%})")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

if __name__ == "__main__":
    main()
