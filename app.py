import streamlit as st
import pandas as pd
import tempfile
from yt_dlp import YoutubeDL
from urllib.parse import urlparse, parse_qs
from extract_and_analyze import download_audio_mp3
from rating_engine import rate_song
import os

def sanitize_youtube_url(input_url):
    parsed = urlparse(input_url)
    query = parse_qs(parsed.query)
    if 'list' in query:
        playlist_id = query['list'][0]
        return f"https://www.youtube.com/playlist?list={playlist_id}"
    else:
        return input_url

def fetch_metadata(url):
    last_exc = None
    cookie_path = 'cookies.txt' if os.path.exists('cookies.txt') else None
    for flat in (True, False):
        ydl_opts = {
            'quiet': True,
            'extract_flat': flat,
            'noplaylist': False,
            'skip_download': True,
            'nocheckcertificate': True,
        }
        if cookie_path:
            ydl_opts['cookiefile'] = cookie_path
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info:
                    return info
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(last_exc or "yt-dlp returned no data")

st.set_page_config(page_title="Qualitune Demo", layout="centered")
st.title("Qualitune - demo")
st.caption("Automated technical analysis for editorial filtering")

input_url = st.text_input("Enter a YouTube URL (playlist or song):", "")

def analyze_url(url):
    info = fetch_metadata(url)

    urls = []
    if info.get('_type') == 'playlist':
        st.success(f"Playlist detected: {info.get('title', 'Untitled')} ({len(info['entries'])} tracks)")
        for entry in info['entries']:
            if entry and 'id' in entry:
                urls.append(f"https://www.youtube.com/watch?v={entry['id']}")
    else:
        st.success("Single song detected")
        urls.append(info['webpage_url'])

    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, track_url in enumerate(urls):
            st.write(f"Analyzing {track_url} ({i+1}/{len(urls)})")
            try:
                mp3_path, title = download_audio_mp3(track_url, tmpdir)
                result = rate_song(mp3_path)
                results.append({
                    "title": title,
                    "rating": result["score"],
                    "issues": ", ".join(result["details"]) if result.get("details") else (result.get("error") or "Unknown"),
                    "url": track_url
                })
            except Exception as e:
                st.warning(f"Skipping: {track_url}\nReason: {str(e)}")
                results.append({
                    "title": "Cannot analyze",
                    "rating": 0,
                    "issues": "Not playable or extractable",
                    "url": track_url
                })

    return pd.DataFrame(results)


if input_url:
    url = sanitize_youtube_url(input_url)

    # Streamlit relanza el script en cada interacción (p. ej. al pulsar
    # "Download"); se guardan los resultados para no volver a descargar y
    # analizar todo.
    if st.session_state.get("analyzed_url") != url:
        with st.spinner("Analyzing…"):
            try:
                st.session_state["results"] = analyze_url(url)
                st.session_state["analyzed_url"] = url
            except Exception as e:
                st.session_state.pop("results", None)
                st.session_state.pop("analyzed_url", None)
                st.error(f"Error during analysis: {e}")

    df = st.session_state.get("results")
    if df is not None and st.session_state.get("analyzed_url") == url:
        st.subheader("Results")
        st.dataframe(df[["title", "rating", "issues"]], use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name="results.csv",
            mime="text/csv"
        )
