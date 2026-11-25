# Qualitune

**Automated technical audio-quality analysis for editorial filtering.**

Paste a YouTube link (a single song or a whole playlist) and Qualitune downloads the audio, measures its technical quality and rates each track from 0 to 5, with a readable list of what is right and what is wrong.

The idea comes from music curation: before listening to dozens of submissions, filter out the ones with obvious technical problems (clipping, noise, poor mastering).

## What it measures

| Area | Metrics | Why it matters |
|---|---|---|
| **Loudness** | Integrated loudness (LUFS), loudness range | Streaming platforms normalise to around −14 LUFS; much quieter tracks sound weak next to others |
| **Peaks** | True peak (dBTP) | Peaks above 0 dBTP distort when the audio is encoded or played back |
| **Dynamics** | Crest factor, clipping ratio | Over-compressed or clipped masters sound flat and distorted |
| **Noise** | Signal-to-noise ratio (dB) | Hiss and background noise from poor recordings |
| **Frequency balance** | Energy per frequency band | Masters with too much or too little bass, mids or treble |
| **Tuning** *(optional)* | Pitch deviation per note, in cents | Out-of-tune vocals (needs Spleeter to isolate the voice) |
| **Duration** | Track length | Flags unusually short or long tracks |

Example output for a heavily clipped track:

```
❌ Loudness out of range (-4.5 LUFS)
❌ True peak above 0 (+2.0 dBTP): inter-sample distortion
❌ Clipping (12.39% of samples)
⚠️ Noticeable background noise (SNR ≈ 22 dB)
```

*(The app itself reports in Spanish.)*

## How to run it

Requirements: Python 3.9+ and [ffmpeg](https://ffmpeg.org/download.html) installed on your system.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional vocal-tuning analysis uses [Spleeter](https://github.com/deezer/spleeter), which is heavy and needs an older Python (3.8–3.10):

```bash
pip install -r requirements-vocals.txt
```

Without it, everything else works and tuning is simply skipped.

### Building a reference dataset

`make feed` reads the YouTube links in `songs.csv`, downloads and analyses each one, and writes the metrics to `reference_dataset.csv`, useful for calibrating the scoring thresholds against reference tracks.

## Project structure

| File | Role |
|---|---|
| `app.py` | Streamlit interface |
| `rating_engine.py` | Turns the metrics into a 0–5 score and the list of findings |
| `analyze_song.py` | Runs every analysis on one audio file |
| `dynamics.py`, `noise.py`, `frequencies.py`, `tuning.py` | One module per family of metrics |
| `extract_and_analyze.py`, `generate_csv_from_playlist.py` | Download from YouTube and batch analysis |

## Built with

Python · [librosa](https://librosa.org/) · [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) · NumPy / SciPy · [yt-dlp](https://github.com/yt-dlp/yt-dlp) · Streamlit

Presented at a Telefónica Open Gateway hackathon (2025) and developed further afterwards.
