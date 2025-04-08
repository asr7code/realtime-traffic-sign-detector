import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import tensorflow as tf
import cv2
import numpy as np
import pickle

# Load your model and label binarizer
model = tf.keras.models.load_model("best_model.h5")
with open("label_binarizer.pkl", "rb") as f:
    lb = pickle.load(f)

st.title("🚦 Real-Time Traffic Sign Detection")

# Webcam transformer
class SignDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        resized = cv2.resize(img, (32, 32))
        normalized = resized / 255.0
        reshaped = np.expand_dims(normalized, axis=0)

        pred = model.predict(reshaped)
        label = lb.classes_[np.argmax(pred)]

        cv2.putText(img, f'{label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img

webrtc_streamer(key="traffic-detect", video_transformer_factory=SignDetector)
