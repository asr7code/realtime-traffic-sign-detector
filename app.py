import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pickle
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from PIL import Image
import av

# Load the trained model and label binarizer
model = tf.keras.models.load_model("best_model.h5")
with open("label_binarizer.pkl", "rb") as f:
    lb = pickle.load(f)

# Voice alert using browser JS (only works online)
def speak(label):
    js_code = f'''
        <script>
            var msg = new SpeechSynthesisUtterance("{label}");
            window.speechSynthesis.speak(msg);
        </script>
    '''
    st.components.v1.html(js_code)

# Title
st.set_page_config(page_title="Real-Time Traffic Sign Detection", layout="centered")
st.title("🚦 Real-Time Traffic Sign Recognition")
st.markdown("Show a traffic sign to your camera and it will recognize and speak the name!")

# Define the video processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_label = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_resized = cv2.resize(img, (30, 30)) / 255.0
        img_expanded = np.expand_dims(img_resized, axis=0)

        preds = model.predict(img_expanded)
        pred_idx = np.argmax(preds)
        label = lb.classes_[pred_idx]

        # Show prediction on the video frame
        cv2.putText(img, f"{label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Only speak if the label changed
        if label != self.last_label:
            self.last_label = label
            speak(label)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start webcam stream
webrtc_streamer(key="traffic-detect", video_processor_factory=VideoProcessor)
