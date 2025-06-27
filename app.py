from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load your model
MODEL_PATH = "pill_model.keras"
model = load_model(MODEL_PATH)

# Define class labels
CLASS_LABELS = ["Amoxicillin 500 MG","Atomoxetine 25 MG","Calcitriol 0.00025 MG","Oseltamivir 45 MG","Ramipril 5 MG","apixaben 2.5 MG","aprepitant 80 MG","benzonatate 100 MG","carvedilol 3.125 MG","celecoxib 200 MG","duloxetine 30 MG","eltrombopag 25 MG","montelukast 10 MG","mycophenolate mofetil 250 MG","pantoprazole 40 MG","pitavastatin 1 MG","prasugrel 10 MG","saxagliptin 5 MG","sitagliptin 50 MG","tadalafil 5 MG"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    # Save and preprocess the image
    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))  # Adjust size as per your model
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class = CLASS_LABELS[np.argmax(prediction)]

    return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
    app.run(debug=True)
