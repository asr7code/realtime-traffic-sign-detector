import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
import pickle
import pyttsx3

# Load model and label binarizer
model = tf.keras.models.load_model("best_model.h5")
with open("label_binarizer.pkl", "rb") as f:
    lb = pickle.load(f)

# Text-to-speech engine
engine = pyttsx3.init()

# Resize and preprocess frame for prediction
def preprocess(frame):
    frame = cv2.resize(frame, (64, 64))
    frame = frame.astype("float") / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

# Predict traffic sign and speak
def predict_and_alert(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocessed = preprocess(image_rgb)
    prediction = model.predict(preprocessed)[0]
    class_index = np.argmax(prediction)
    class_label = lb.classes_[class_index]

    # Voice alert
    engine.say(f"Traffic Sign: {class_label}")
    engine.runAndWait()

    # Return label for display
    return f"Detected: {class_label}"

# Gradio Interface
iface = gr.Interface(
    fn=predict_and_alert,
    inputs=gr.Image(source="webcam", streaming=True),
    outputs="text",
    title="Real-Time Traffic Sign Detector with Voice Alert",
    live=True,
)

if __name__ == "__main__":
    iface.launch()
