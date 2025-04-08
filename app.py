import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pickle
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load model and label binarizer
@st.cache_resource
def load_model_and_label_binarizer():
    model = tf.keras.models.load_model("best_model.h5", compile=False)
    with open("label_binarizer.pkl", "rb") as f:
        lb = pickle.load(f)
    return model, lb

model, lb = load_model_and_label_binarizer()

# Title
st.title("🚦 Real-Time Traffic Sign Detector")
st.markdown("Using your webcam to detect traffic signs in real-time!")

# Custom VideoTransformer for streamlit-webrtc
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_resized = cv2.resize(img, (64, 64))
        img_normalized = img_resized.astype("float32") / 255.0
        input_img = np.expand_dims(img_normalized, axis=0)

        # Predict
        predictions = model.predict(input_img)
        predicted_class = lb.classes_[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        # Add prediction to frame
        label = f"{predicted_class} ({confidence:.2f}%)"
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img

# Start webcam stream
webrtc_streamer(
    key="traffic-sign-detection",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
