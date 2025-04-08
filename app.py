import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import tensorflow as tf
import numpy as np
import cv2
import pickle

# Set Streamlit configuration first
st.set_page_config(page_title="Real-Time Traffic Sign Detector", layout="centered")
st.title("🚦 Real-Time Traffic Sign Recognition")
st.markdown("Show a traffic sign in front of your webcam and it will recognize it!")

# Load the trained model from the .keras file with compile=False
model = tf.keras.models.load_model("best_model.keras", compile=False)

# Load the label binarizer
with open("label_binarizer.pkl", "rb") as f:
    lb = pickle.load(f)

# Define a video transformer that processes each frame
class TrafficSignDetector(VideoTransformerBase):
    def __init__(self):
        self.result = None
        self.last_label = None

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # Adjust image size if needed (e.g., if your model was trained with a different size than (30,30))
        img_resized = cv2.resize(img, (30, 30))
        img_array = img_resized.astype("float32") / 255.0
        img_input = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_input)
        pred_idx = int(np.argmax(prediction))
        label = lb.classes_[pred_idx]
        confidence = prediction[0][pred_idx]

        self.result = f"{label} ({confidence*100:.2f}%)"

        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if label != self.last_label:
            self.last_label = label
            st.components.v1.html(f"""
                <script>
                    var msg = new SpeechSynthesisUtterance("Caution! {label}");
                    msg.lang = "en-US";
                    msg.rate = 0.9;
                    window.speechSynthesis.speak(msg);
                </script>
            """, height=0)

        return img

webrtc_ctx = webrtc_streamer(
    key="traffic-sign-detect",
    video_transformer_factory=TrafficSignDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)

if webrtc_ctx.video_transformer and webrtc_ctx.video_transformer.result:
    st.success(f"Detected Sign: {webrtc_ctx.video_transformer.result}")
