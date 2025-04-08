import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import tensorflow as tf
import numpy as np
import cv2
import pickle
from PIL import Image

# Load the trained model and label binarizer
model = tf.keras.models.load_model("best_model.keras")

with open("label_binarizer.pkl", "rb") as f:
    lb = pickle.load(f)

# Streamlit page config
st.set_page_config(page_title="Real-Time Traffic Sign Detector", layout="centered")
st.title("🚦 Real-Time Traffic Sign Recognition")
st.markdown("Show a traffic sign in front of your webcam and it will recognize it!")

# Define Video Transformer
class TrafficSignDetector(VideoTransformerBase):
    def __init__(self):
        self.result = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_resized = cv2.resize(img, (30, 30))  # Resize to model input
        img_array = np.expand_dims(img_resized, axis=0) / 255.0

        # Predict
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        class_name = lb.classes_[class_index]
        confidence = prediction[0][class_index]

        # Save result to show below
        self.result = f"{class_name} ({confidence*100:.2f}%)"

        # Draw on frame
        cv2.putText(img, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

# Initialize and start webcam
webrtc_ctx = webrtc_streamer(
    key="example",
    video_transformer_factory=TrafficSignDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)

# Show results if available
if webrtc_ctx.video_transformer:
    result = webrtc_ctx.video_transformer.result
    if result:
        st.success(f"Detected Sign: **{result}**")
