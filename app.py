from flask import Flask, render_template, request
import pandas as pd
import os
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CSV load
CSV_PATH = os.path.join(BASE_DIR, "labels.csv")
df = pd.read_csv(CSV_PATH)

# Use pretrained model (NO .h5 needed 🔥)
model = MobileNetV2(weights="imagenet")

def predict_breed(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=1)[0][0]

    breed = decoded[1]
    confidence = decoded[2] * 100  # percentage

    return breed, confidence  # breed name

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    if file:
        if not os.path.exists('static'):
            os.makedirs('static')

        filepath = os.path.join('static', file.filename)
        file.save(filepath)

        # ⭐ FIX HERE
        breed, confidence = predict_breed(filepath)

        return render_template(
            'index.html',
            prediction=breed,
            confidence=round(confidence, 2),
            img=filepath
        )

    return "No file uploaded"

if __name__ == '__main__':
    app.run(debug=True)