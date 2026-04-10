from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
import os
import tempfile
from tensorflow.keras.models import load_model

app = Flask(__name__)

# 🔥 SAFE MODEL LOADING (prevents crash)
MODEL_PATH = "model/cnn_cough_sr22050_v2.h5"

try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)
    model = None   # Prevent app crash

# Constants
SAMPLE_RATE = 22050
DURATION = 1.5
N_MFCC = 40

# Feature extraction
def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )
    return mfcc

# Silence detection
def is_silent(audio, threshold=0.01):
    return np.mean(np.abs(audio)) < threshold

# Home route
@app.route("/")
def index():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    # 🔥 Check if model loaded
    if model is None:
        return jsonify({
            "error": "Model failed to load"
        }), 500

    if "audio" not in request.files:
        return jsonify({"error": "No audio received"}), 400

    audio_file = request.files["audio"]

    # Save temp file
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = temp.name
    temp.close()

    try:
        audio_file.save(temp_path)

        audio, _ = librosa.load(
            temp_path,
            sr=SAMPLE_RATE,
            duration=DURATION
        )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Silence check
    if is_silent(audio):
        return jsonify({
            "prediction": "No Cough",
            "confidence": 0.0
        })

    # MFCC
    mfcc = extract_mfcc(audio)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    # Prediction
    prob = float(model.predict(mfcc, verbose=0)[0][0])
    result = "Cough" if prob > 0.6 else "No Cough"

    return jsonify({
        "prediction": result,
        "confidence": round(prob, 3)
    })


# 🔥 IMPORTANT FOR RENDER (PORT FIX)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)