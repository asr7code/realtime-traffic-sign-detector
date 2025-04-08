from flask import Flask, render_template, request, jsonify
import base64
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from io import BytesIO
import re

app = Flask(__name__)

model = load_model('model/best_model.h5')
with open('model/label_binarizer.pkl', 'rb') as f:
    lb = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']
    image_data = re.sub('^data:image/.+;base64,', '', image_data)
    image = base64.b64decode(image_data)
    nparr = np.frombuffer(image, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resize and preprocess
    resized = cv2.resize(frame, (32, 32))
    resized = resized.astype('float32') / 255.0
    resized = np.expand_dims(resized, axis=0)

    preds = model.predict(resized)
    label = lb.classes_[np.argmax(preds)]

    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(debug=True)
