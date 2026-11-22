import numpy as np
import librosa


def calculate_snr_db(y, sr, n_fft=4096, percentile=10, freq_range=(50, 15000), max_db=90.0):
    """
    Estima la relación señal/ruido (dB) frente a un ruido de fondo de banda
    ancha (hiss, ruido de grabación) mediante "mínimos estadísticos":

    - Para cada frecuencia se toma su potencia en los momentos más silenciosos
      (percentil bajo en el tiempo).
    - El ruido de fondo es la mediana de esos mínimos entre frecuencias: las
      frecuencias ocupadas por la música son minoría y no lo sesgan, mientras
      que el ruido de banda ancha está en todas.
    - La señal es la potencia media total.

    Valores altos = canción limpia. Se ignoran silencios y fundidos para que
    no cuenten como ruido. No detecta zumbidos de banda estrecha (50 Hz, etc.).
    """
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=n_fft // 4)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    S = S[(freqs >= freq_range[0]) & (freqs <= freq_range[1])]

    frame_db = librosa.power_to_db(S.sum(axis=0), ref=np.max)
    S = S[:, frame_db > -60]
    if S.size == 0:
        return np.nan

    floor = np.median(np.percentile(S, percentile, axis=1)) * S.shape[0]
    signal = S.mean(axis=1).sum()
    if floor <= 0:
        return max_db
    return float(min(10 * np.log10(signal / floor), max_db))
