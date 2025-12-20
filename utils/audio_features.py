

import librosa
import numpy as np

# Audio parameters (must match training)
SAMPLE_RATE = 22050
N_MFCC = 40

def is_silent(audio, threshold=1e-4):
    """
    Check if audio is mostly silence using energy.
    """
    energy = np.mean(audio ** 2)
    return energy < threshold

def extract_mfcc(audio):
    """
    Extract MFCC features from audio for CNN input.
    """
    # Remove silence
    audio, _ = librosa.effects.trim(audio, top_db=25)

    # Handle very short audio
    if len(audio) < 1000:
        audio = np.pad(audio, (0, 1000 - len(audio)))

    # MFCC extraction
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=1024,
        hop_length=512
    )

    # Fix time dimension for CNN
    mfcc = librosa.util.fix_length(mfcc, size=44, axis=1)

    # Normalize
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

    return mfcc
