import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyttsx3
import pickle

# Load model
model = load_model("best_model.h5")

# Load label binarizer
with open("label_binarizer.pkl", "rb") as f:
    lb = pickle.load(f)

class_names = lb.classes_

# Text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def preprocess(frame):
    frame_resized = cv2.resize(frame, (64, 64))
    frame_normalized = frame_resized.astype("float32") / 255.0
    return np.expand_dims(frame_normalized, axis=0)

cap = cv2.VideoCapture(0)

last_prediction = None
confidence_threshold = 0.85

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_frame = preprocess(frame)
    predictions = model.predict(input_frame)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]

    if confidence > confidence_threshold:
        label = class_names[predicted_class]
        cv2.putText(frame, f"{label} ({confidence*100:.2f}%)", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if last_prediction != label:
            print(f"Detected: {label} ({confidence*100:.2f}%)")
            speak(f"Traffic Sign: {label}")
            last_prediction = label

    cv2.imshow("Traffic Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
