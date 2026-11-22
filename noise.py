import numpy as np
from scipy.signal import find_peaks
import librosa

def calculate_noise_ratio(y, sr, n_fft=16384, min_peak_height=0.05, freq_range=(50, 16000)):
    # Espectro medio de toda la canción: STFT con ventana Hann y promedio
    # de magnitudes en el tiempo (antes solo se analizaban las primeras
    # n_fft muestras, unos 0,37 s, porque rfft(n=n_fft) recorta la señal)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=n_fft // 2, window='hann'))
    spectrum = S.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Limitar al rango de frecuencias relevante
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    spectrum = spectrum[freq_mask]
    freqs = freqs[freq_mask]
    
    # Detectar picos armónicos
    peaks, _ = find_peaks(spectrum, height=np.max(spectrum)*min_peak_height, distance=10)
    
    # Crear máscara para componentes armónicos (picos + vecindario)
    harmonic_mask = np.zeros_like(spectrum, dtype=bool)
    peak_width = 8  # bins de frecuencia a cada lado del pico
    for p in peaks:
        start = max(0, p - peak_width)
        end = min(len(spectrum), p + peak_width)
        harmonic_mask[start:end] = True
    
    # Calcular energías
    harmonic_energy = np.sum(spectrum[harmonic_mask])
    noise_energy = np.sum(spectrum[~harmonic_mask])
    total_energy = harmonic_energy + noise_energy
    
    # Evitar división por cero
    if total_energy == 0:
        return 0.0
    
    return noise_energy / total_energy