# Cough Detection Web Application

A Flask-based web application that detects whether an audio file contains
a cough sound using a trained deep learning (CNN) model.

---

## Features
- Upload cough audio and get instant prediction
- Audio feature extraction using MFCC
- CNN-based cough classification
- Simple web interface using Flask

---

## Tech Stack
- Python
- Flask
- TensorFlow (CPU)
- Librosa
- NumPy
- SciPy
- HTML / CSS

---

## Project Structure

```
cough-detection/
│
├── app.py # Main Flask application (entry point)
├── requirements.txt # Project dependencies
│
├── model/
│ └── cnn_cough_sr22050_v2.h5 # Trained CNN model
│
├── templates/
│ └── index.html # Frontend UI
│
├── utils/
│ ├── audio_features.py # Audio preprocessing and MFCC extraction
│ └── pycache/

```

---

## Requirements
- Python 3.8 or higher
- pip (Python package manager)

---

## Installation

### Clone the repository
```bash
git clone https://github.com/your-username/cough-detection.git
cd cough-detection
```  

### Install dependencies
```bash
pip install -r requirements.txt
```  

## Run the Application
From the project root directory:

```bash
python app.py
```  

```text
Running on http://127.0.0.1:5000/
```

## Open in Browser
Open your browser and go to:

http://127.0.0.1:5000/

---


