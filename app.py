import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pickle
from PIL import Image
import io

st.set_page_config(page_title="Traffic Sign Detector", layout="centered")

# Suppress TensorFlow warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Load the trained model and label binarizer
@st.cache_resource
def load_model_and_labels():
    try:
        model = tf.keras.models.load_model("best_model.h5", compile=False)
        with open("label_binarizer.pkl", "rb") as f:
            lb = pickle.load(f)
        return model, lb
    except Exception as e:
        st.error(f"Error loading model or labels: {e}")
        return None, None

model, label_binarizer = load_model_and_labels()

st.title("🚦 Real-Time Traffic Sign Classifier")

uploaded_file = st.file_uploader("Upload a traffic sign image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for model
    img_array = np.array(image.convert("RGB"))
    img_resized = cv2.resize(img_array, (64, 64))
    img_normalized = img_resized.astype("float32") / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)

    if model and label_binarizer:
        try:
            predictions = model.predict(img_expanded)
            predicted_label = label_binarizer.classes_[np.argmax(predictions)]
            confidence = np.max(predictions) * 100

            st.success(f"Prediction: **{predicted_label}** with **{confidence:.2f}%** confidence.")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.error("Model or label binarizer failed to load.")
else:
    st.info("Upload a JPG or PNG image of a traffic sign to classify it.")
