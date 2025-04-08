import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import tensorflow as tf
import numpy as np
import cv2
import pickle

# Set Streamlit page configuration (must be the first command)
st.set_page_config(page_title="Real-Time Traffic Sign Detector", layout="centered")
st.title("🚦 Real-Time Traffic Sign Recognition")
st.markdown("Show a traffic sign in front of your webcam and it will recognize it!")

# Load the trained model from the .keras file (using compile=False to avoid optimizer issues)
model = tf.keras.models.load_model("best_model.keras", compile=False)

# Load the label binarizer
with open("label_binarizer.pkl", "rb") as f:
    lb = pickle.load(f)

# Define a video transformer to process webcam frames
class TrafficSignDetector(VideoTransformerBase):
    def __init__(self):
        self.result = None
        self.last_label = None

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert the frame to a NumPy array in BGR format
        img = frame.to_ndarray(format="bgr24")
        # Resize image to match model input; adjust dimensions if needed.
        # (Using 30x30 as in previous code—if your model was trained on 64x64, update this accordingly.)
        img_resized = cv2.resize(img, (30, 30))
        # Normalize the pixel values
        img_array = img_resized.astype("float32") / 255.0
        # Add batch dimension
        img_input = np.expand_dims(img_array, axis=0)
        
        # Predict using the model
        prediction = model.predict(img_input)
        pred_idx = int(np.argmax(prediction))
        # Use the label binarizer to get the traffic sign name
        label = lb.classes_[pred_idx]
        confidence = prediction[0][pred_idx]
        
        # Save the result to display later
        self.result = f"{label} ({confidence*100:.2f}%)"

        # Draw the label on the frame
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Use browser-based voice alert via JavaScript if the label changes
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

# Start the webcam stream using streamlit-webrtc
webrtc_ctx = webrtc_streamer(
    key="traffic-sign-detect",
    video_transformer_factory=TrafficSignDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)

# If available, display the detected result text below the video stream.
if webrtc_ctx.video_transformer and webrtc_ctx.video_transformer.result:
    st.success(f"Detected Sign: {webrtc_ctx.video_transformer.result}")
