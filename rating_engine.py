from analyze_song import analyze_song
import numpy as np


def _rate_dynamics(dyn, details):
    """Dinámica: 20 puntos (loudness, true peak, crest factor y clipping, 5 cada uno)."""
    score = 0

    # Loudness integrado. Spotify/YouTube normalizan a unos -14 LUFS, así que
    # no se premia "más fuerte": se penalizan los extremos en ambos sentidos.
    lufs = dyn.get('lufs', np.nan)
    if np.isnan(lufs) or np.isinf(lufs):
        details.append("⚠️ Loudness no medible")
    elif -16 <= lufs <= -7:
        score += 5
        details.append(f"✅ Loudness en rango habitual ({lufs:.1f} LUFS)")
    elif -20 <= lufs < -16:
        score += 3
        details.append(f"⚠️ Loudness algo bajo ({lufs:.1f} LUFS), suena flojo frente a otros temas")
    elif -7 < lufs <= -5:
        score += 3
        details.append(f"⚠️ Loudness muy alto ({lufs:.1f} LUFS), las plataformas lo bajarán y probablemente está sobrecomprimido")
    else:
        details.append(f"❌ Loudness fuera de rango ({lufs:.1f} LUFS)")

    # True peak: por encima de -1 dBTP hay riesgo de distorsión al codificar a MP3/AAC
    tp = dyn.get('true_peak_db', np.nan)
    if np.isnan(tp):
        details.append("⚠️ True peak no medible")
    elif tp <= -1:
        score += 5
        details.append(f"✅ True peak seguro ({tp:.1f} dBTP)")
    elif tp <= 0:
        score += 3
        details.append(f"⚠️ True peak justo ({tp:.1f} dBTP), puede distorsionar al codificar")
    else:
        details.append(f"❌ True peak por encima de 0 ({tp:+.1f} dBTP), distorsión entre muestras")

    # Crest factor (pico/RMS en dB): bajo = aplastado por el limitador; alto = sin masterizar
    crest = dyn.get('crest_factor_db', np.nan)
    if np.isnan(crest):
        details.append("⚠️ Crest factor no medible")
    elif 7 <= crest <= 14:
        score += 5
        details.append(f"✅ Dinámica sana (crest factor {crest:.1f} dB)")
    elif 14 < crest <= 18:
        score += 3
        details.append(f"⚠️ Muy dinámico (crest factor {crest:.1f} dB), posible falta de masterización")
    elif 5 <= crest < 7:
        score += 2
        details.append(f"⚠️ Dinámica muy comprimida (crest factor {crest:.1f} dB)")
    else:
        details.append(f"❌ Crest factor extremo ({crest:.1f} dB)")

    # Clipping (techos planos)
    clipping = dyn.get('clipping_ratio', np.nan)
    if np.isnan(clipping):
        details.append("⚠️ Clipping no medible")
    elif clipping < 0.0001:
        score += 5
        details.append("✅ Sin clipping detectable")
    elif clipping < 0.001:
        score += 3
        details.append(f"⚠️ Clipping puntual ({clipping*100:.3f}% de las muestras)")
    else:
        details.append(f"❌ Clipping ({clipping*100:.2f}% de las muestras)")

    return score


def _rate_noise(snr, details):
    """Ruido de fondo de banda ancha: 20 puntos."""
    if snr >= 40:
        details.append(f"✅ Ruido de fondo inapreciable (SNR ≈ {snr:.0f} dB)")
        return 20
    if snr >= 30:
        details.append(f"⚠️ Ruido de fondo detectable en partes suaves (SNR ≈ {snr:.0f} dB)")
        return 15
    if snr >= 20:
        details.append(f"⚠️ Ruido de fondo notable (SNR ≈ {snr:.0f} dB)")
        return 10
    details.append(f"❌ Ruido de fondo excesivo (SNR ≈ {snr:.0f} dB)")
    return 0


def _rate_tuning(cents, details):
    """Afinación vocal: 20 puntos. cents = desviación mediana respecto al semitono."""
    if cents <= 12:
        details.append(f"✅ Afinación vocal precisa (±{cents:.0f} cents)")
        return 20
    if cents <= 20:
        details.append(f"⚠️ Afinación vocal aceptable (±{cents:.0f} cents)")
        return 15
    if cents <= 30:
        details.append(f"⚠️ Afinación vocal irregular (±{cents:.0f} cents)")
        return 8
    details.append(f"❌ Voz desafinada (±{cents:.0f} cents)")
    return 0


def _rate_bands(bands, details):
    """Balance de frecuencias: 20 puntos. Umbrales provisionales, pendientes de calibrar."""
    total = sum(bands.values())
    if total <= 0:
        details.append("⚠️ Balance de frecuencias no medible")
        return 0
    lows = (bands['sub_bass'] + bands['bass']) / total
    mids = (bands['low_mids'] + bands['mids']) / total
    highs = (bands['highs'] + bands['air']) / total

    if 0.2 < lows < 0.4 and 0.3 < mids < 0.5 and 0.2 < highs < 0.4:
        details.append("✅ Balance de frecuencias ideal")
        return 20
    if abs(lows - 0.3) < 0.15 and abs(mids - 0.4) < 0.15 and abs(highs - 0.3) < 0.15:
        details.append("⚠️ Balance de frecuencias aceptable")
        return 15
    details.append("❌ Desbalance frecuencial detectado")
    return 0


def rate_song(filepath):
    """
    Evalúa la calidad técnica de una canción basándose en múltiples métricas.
    Devuelve una nota de 0 a 5 y un análisis detallado. Las secciones que no
    se pueden medir (p. ej. afinación en un instrumental) no cuentan ni a favor
    ni en contra.
    """
    try:
        analysis = analyze_song(filepath)
        if analysis.get('error'):
            return {
                'score': 0,
                'error': analysis['error'],
                'details': None
            }

        score = 0
        max_score = 0
        details = []

        # 1. Dinámica
        dynamics = analysis.get('dynamics')
        if dynamics:
            max_score += 20
            score += _rate_dynamics(dynamics, details)

        # 2. Ruido
        snr = analysis.get('snr_db', np.nan)
        if snr is not None and not np.isnan(snr):
            max_score += 20
            score += _rate_noise(snr, details)

        # 3. Afinación vocal (solo si hay voz medible)
        tuning = analysis.get('tuning') or {}
        cents = tuning.get('cents_dev', np.nan)
        if cents is not None and not np.isnan(cents):
            max_score += 20
            score += _rate_tuning(cents, details)
        else:
            details.append("ℹ️ Afinación no evaluada (sin voz detectable o separación no disponible)")

        # 4. Balance de frecuencias
        bands = analysis.get('frequency_bands')
        if bands:
            max_score += 20
            score += _rate_bands(bands, details)

        # 5. Duración
        max_score += 20
        duration = int(analysis.get('duration') or 0)
        mins, secs = divmod(duration, 60)
        if 120 <= duration <= 600:
            score += 20
            details.append(f"✅ Duración adecuada ({mins}:{secs:02d})")
        else:
            details.append(f"⚠️ Duración atípica ({mins}:{secs:02d})")

        final_score = round((score / max_score) * 5, 2) if max_score > 0 else 0.0

        return {
            'score': final_score,
            'details': details,
            'raw_analysis': analysis
        }

    except Exception as e:
        return {
            'score': 0,
            'error': str(e),
            'details': None
        }
