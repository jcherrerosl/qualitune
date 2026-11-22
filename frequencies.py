import librosa
import numpy as np

BANDS = {
    "sub_bass": (20, 60),
    "bass": (60, 120),
    "low_mids": (120, 300),
    "mids": (300, 2000),
    "highs": (2000, 7500),
    "air": (7500, 15000),
}


def analyze_bands(y, sr):
    try:
        # Una sola STFT para todas las bandas (antes se calculaba 6 veces)
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        result = {}
        for name, (fmin, fmax) in BANDS.items():
            band = (freqs >= fmin) & (freqs <= fmax)
            result[name] = round(float(np.mean(S[band, :])), 4) if np.any(band) else 0.0
        return result

    except Exception as e:
        raise RuntimeError(f"Error analyzing frequency bands: {e}")
