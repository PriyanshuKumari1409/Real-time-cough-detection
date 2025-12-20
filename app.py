# Run app
from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
import os
import tempfile
from tensorflow.keras.models import load_model

from utils.audio_features import extract_mfcc, is_silent

# ---------------------------
# Flask app initialization
# ---------------------------
app = Flask(__name__)

# ---------------------------
# Load CNN model ONCE
# ---------------------------
MODEL_PATH = "model/cnn_cough_sr22050_v2.h5"
model = load_model(MODEL_PATH)

# ---------------------------
# Home page
# ---------------------------
@app.route("/")
def index():
    return render_template("index.html")

# ---------------------------
# Prediction endpoint
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "audio" not in request.files:
        return jsonify({"error": "No audio received"}), 400

    audio_file = request.files["audio"]

    # ✅ Windows-safe temporary file
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = temp.name
    temp.close()

    try:
        audio_file.save(temp_path)
        audio, _ = librosa.load(temp_path, sr=22050, duration=1.0)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # -------- Silence gate --------
    if is_silent(audio):
        return jsonify({
            "prediction": "No Cough",
            "confidence": 0.0
        })

    # -------- Feature extraction --------
    mfcc = extract_mfcc(audio)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    # -------- CNN inference --------
    prob = float(model.predict(mfcc, verbose=0)[0][0])
    result = "Cough" if prob > 0.6 else "No Cough"

    return jsonify({
        "prediction": result,
        "confidence": round(prob, 3)
    })

# ---------------------------
# Run app
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
