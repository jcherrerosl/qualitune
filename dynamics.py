import librosa
import numpy as np
import pyloudnorm as pyln
from scipy.signal import resample_poly


def clipped_ratio(y, peak, min_run=4, tol=1e-5):
    """
    Fracción de muestras en "techos planos": tramos de min_run o más muestras
    seguidas prácticamente idénticas y cerca del pico, que es la forma de una
    señal recortada. Un pico normal (aunque llegue a 0 dBFS) es curvo y no
    cuenta. Tras decodificar un MP3 los techos dejan de ser planos, así que en
    ese caso el true peak es el mejor indicador.
    """
    if peak <= 0 or len(y) < min_run:
        return 0.0
    flat = (np.abs(np.diff(y)) < tol) & (np.abs(y[1:]) >= 0.95 * peak)
    padded = np.concatenate(([0], flat.astype(np.int8), [0]))
    edges = np.diff(padded)
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    lengths = ends - starts
    clipped = np.sum(lengths[lengths >= min_run - 1] + 1)
    return float(clipped / len(y))


def analyze_dynamics(y, sr):
    try:
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        rms_total = np.sqrt(np.mean(y ** 2))
        peak = np.max(np.abs(y))
        # True peak aproximado: sobremuestreo x4 (como pide ITU-R BS.1770) para
        # detectar picos entre muestras que aparecen al convertir a MP3/AAC
        true_peak = np.max(np.abs(resample_poly(y, 4, 1)))
        true_peak_db = 20 * np.log10(true_peak) if true_peak > 0 else -np.inf
        crest_factor = peak / rms_total if rms_total > 0 else 0
        clipping = clipped_ratio(y, peak)

        return {
            "lufs": round(float(loudness), 2),
            "rms": round(float(rms_total), 6),
            "true_peak_db": round(float(true_peak_db), 2),
            "crest_factor": round(float(crest_factor), 2),
            "crest_factor_db": round(float(20 * np.log10(crest_factor)) if crest_factor > 0 else 0.0, 2),
            "clipping_ratio": round(float(clipping), 6),
        }

    except Exception as e:
        raise RuntimeError(f"Error analyzing dynamics: {e}")
