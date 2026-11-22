import librosa
import numpy as np
import shutil
import soundfile as sf
import tempfile
import os

_separator = None


def _get_separator():
    """Carga el modelo de Spleeter una sola vez (antes se cargaba en cada canción)."""
    global _separator
    if _separator is None:
        from spleeter.separator import Separator  # import diferido: Spleeter es opcional y pesado
        _separator = Separator('spleeter:2stems')
    return _separator


def separate_audio(y, sr):
    try:
        # Baja a 22050 Hz
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=22050)

        # Archivo temporal .wav
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
            temp_path = tmpfile.name
            sf.write(temp_path, y_resampled, 22050)

        output_dir = os.path.dirname(temp_path)
        separator = _get_separator()
        separator.separate_to_file(temp_path, output_dir)

        vocal_path = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(temp_path))[0],
            'vocals.wav'
        )

        vocals, _ = librosa.load(vocal_path, sr=22050)

        # Limpieza: Spleeter 2stems genera vocals.wav y accompaniment.wav,
        # así que se borra la carpeta entera (os.rmdir fallaba al no estar vacía)
        os.remove(temp_path)
        shutil.rmtree(os.path.dirname(vocal_path), ignore_errors=True)

        return vocals, 22050

    except Exception as e:
        print(f"[ERROR] Error al separar audio: {e}")
        return None, sr



def _note_segments(midi, voiced, min_frames):
    """Trocea el contorno de tono en notas: tramos con voz sin saltos de más de medio semitono."""
    segments, start = [], None
    for k in range(len(midi)):
        is_break = (not voiced[k]) or (start is not None and abs(midi[k] - midi[k - 1]) > 0.5)
        if is_break:
            if start is not None and k - start >= min_frames:
                segments.append((start, k))
            start = k if voiced[k] else None
        elif start is None:
            start = k
    if start is not None and len(midi) - start >= min_frames:
        segments.append((start, len(midi)))
    return segments


def check_tune(vocals_mono, sr, min_voiced_ratio=0.1, min_note_s=0.1, hop_length=512):
    """
    Mide la afinación de la voz separada.

    Devuelve un dict con:
    - pitch: nota mediana aproximada
    - voiced_ratio: fracción de la canción con voz detectada (0-1)
    - n_notes: número de notas analizadas
    - cents_dev: desviación mediana (en cents) del tono de cada nota respecto
      al semitono más cercano, tras descontar la afinación global (una canción
      afinada a 432 Hz no cuenta como desafinada). Cada nota se resume por su
      tono mediano, así que el vibrato no penaliza. 0 = perfecta; 100 cents =
      un semitono. Es NaN si no hay voz suficiente.
    """
    f0, voiced_flag, _ = librosa.pyin(
        vocals_mono,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        hop_length=hop_length,
    )

    voiced = voiced_flag & ~np.isnan(f0)
    voiced_ratio = float(np.mean(voiced)) if len(voiced) else 0.0
    empty = {'pitch': None, 'voiced_ratio': round(voiced_ratio, 3), 'n_notes': 0, 'cents_dev': np.nan}
    if voiced_ratio < min_voiced_ratio:
        return empty

    midi = np.where(voiced, librosa.hz_to_midi(np.where(voiced, f0, 1.0)), np.nan)
    min_frames = max(1, int(min_note_s * sr / hop_length))
    segments = _note_segments(midi, voiced, min_frames)
    if len(segments) < 3:
        return empty

    note_pitch = np.array([np.median(midi[a:b]) for a, b in segments])
    dev = (note_pitch - np.round(note_pitch)) * 100          # -50..50 cents
    # Afinación global = media circular de las desviaciones
    offset = np.angle(np.mean(np.exp(1j * dev / 100 * 2 * np.pi))) / (2 * np.pi) * 100
    rel = (dev - offset + 50) % 100 - 50

    return {
        'pitch': librosa.midi_to_note(float(np.median(note_pitch))),
        'voiced_ratio': round(voiced_ratio, 3),
        'n_notes': len(segments),
        'cents_dev': round(float(np.median(np.abs(rel))), 1),
    }
