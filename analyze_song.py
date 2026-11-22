from frequencies import analyze_bands
from dynamics import analyze_dynamics
from noise import calculate_snr_db
from tuning import separate_audio, check_tune
import librosa
import numpy as np

def analyze_song(filepath, target_sr=None):
    # target_sr=None conserva la frecuencia original: resamplear deforma los
    # picos y los techos planos del clipping
    results = {}

    try:
        y, sr = librosa.load(filepath, sr=target_sr, mono=True)
    except Exception as e:
        return {
            'error': f"Error loading audio: {e}",
            'frequency_bands': None,
            'dynamics': None,
            'snr_db': np.nan,
            'tuning': {'pitch': None, 'voiced_ratio': np.nan, 'cents_dev': np.nan},
            'duration': 0,
        }

    results['duration'] = len(y) / sr

    # Frecuencias
    try:
        results['frequency_bands'] = analyze_bands(y, sr)
    except Exception as e:
        results['frequency_bands'] = None

    # Dinámica
    try:
        results['dynamics'] = analyze_dynamics(y, sr)
    except Exception as e:
        results['dynamics'] = None

    # Ruido
    try:
        results['snr_db'] = calculate_snr_db(y, sr)
    except Exception as e:
        results['snr_db'] = np.nan

    # Afinación vocal (requiere Spleeter; si no está, se omite)
    empty_tuning = {'pitch': None, 'voiced_ratio': np.nan, 'cents_dev': np.nan}
    try:
        vocals, sr_vocals = separate_audio(y, sr)
        results['tuning'] = check_tune(vocals, sr_vocals) if vocals is not None else empty_tuning
    except Exception as e:
        results['tuning'] = empty_tuning

    results['error'] = None
    return results
