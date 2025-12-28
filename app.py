from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
import os
import tempfile
from tensorflow.keras.models import load_model


app = Flask(__name__)


MODEL_PATH = "model/cnn_cough_sr22050_v2.h5"
model = load_model(MODEL_PATH)

SAMPLE_RATE = 22050
DURATION = 1.5       
N_MFCC = 40


def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )
    return mfcc

def is_silent(audio, threshold=0.01):
    return np.mean(np.abs(audio)) < threshold


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    if "audio" not in request.files:
        return jsonify({"error": "No audio received"}), 400

    audio_file = request.files["audio"]

  
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

    
    if is_silent(audio):
        return jsonify({
            "prediction": "No Cough",
            "confidence": 0.0
        })

   
    mfcc = extract_mfcc(audio)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # (1, n_mfcc, time, 1)

   
    prob = float(model.predict(mfcc, verbose=0)[0][0])

    result = "Cough" if prob > 0.6 else "No Cough"

    return jsonify({
        "prediction": result,
        "confidence": round(prob, 3)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
