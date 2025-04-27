# =============================================================================
# Early environment fixes for Mac M2 Segfaults
# =============================================================================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# =============================================================================
# RAISE – Conversational Analysis App (app.py)
# =============================================================================
# Phase 1: Full Automation + Metrics + Export + Interactive UI + Enhancements
# =============================================================================

import sys
import threading
import shutil
import tempfile
import uuid
import warnings
import logging
import zipfile
import wave
import json

from datetime import datetime, timedelta
from collections import Counter

import numpy as np
import soundfile as sf
import librosa
import torch
import whisper
import whisper.audio as whisper_audio
import whisper.model
import torchaudio
from torchaudio.transforms import Resample

from scipy.signal import butter, filtfilt
from pydub import AudioSegment
from dotenv import load_dotenv
import streamlit as st
from transformers import pipeline
from pyannote.audio import Pipeline as DiarizationPipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import openai
import openai.error

# =============================================================================
# Configure logging
# =============================================================================
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Utility: normalize & low-pass filter
# =============================================================================
def normalize_audio_array(audio: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(audio))
    return audio / max_val if max_val > 0 else audio

def lowpass_filter(audio: np.ndarray, sr: int,
                   cutoff: int = 4000, order: int = 5) -> np.ndarray:
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, audio)

# =============================================================================
# Monkey-patch shutil.copy & os.symlink for speechbrain permissions
# =============================================================================
_orig_shutil_copy = shutil.copy
def _safe_copy(src, dst):
    try:
        return _orig_shutil_copy(src, dst)
    except (PermissionError, OSError) as e:
        warnings.warn(f"[RAISE Patch] Skipped copy: {e}")
shutil.copy = _safe_copy

import os as _os
_orig_symlink = _os.symlink
def _safe_symlink(src, dst):
    try:
        return _orig_symlink(src, dst)
    except (PermissionError, OSError) as e:
        warnings.warn(f"[RAISE Patch] Skipped symlink: {e}")
_os.symlink = _safe_symlink

# =============================================================================
# Ensure np.NaN alias for NumPy 2.x
# =============================================================================
if not hasattr(np, "NaN"):
    np.NaN = np.nan

# =============================================================================
# Whisper patches for NumPy subclass handling
# =============================================================================
_orig_from_numpy = torch.from_numpy
torch.from_numpy = lambda x: _orig_from_numpy(
    np.ascontiguousarray(x.astype(np.float32)).view(np.ndarray)
)
whisper.model.Whisper.set_alignment_heads = lambda self, alignment_heads=None: \
    logger.warning("Skipping alignment patch")
_orig_log_mel = whisper_audio.log_mel_spectrogram
whisper_audio.log_mel_spectrogram = lambda audio, *a, **k: _orig_log_mel(
    np.ascontiguousarray(audio.astype(np.float32)).view(np.ndarray), *a, **k
)
_orig_load = whisper_audio.load_audio
def _safe_load(path: str, sr: int = 16000):
    y, sr0 = librosa.load(path, sr=None, mono=False)
    if sr0 != sr:
        y = librosa.resample(y, orig_sr=sr0, target_sr=sr)
    return np.ascontiguousarray(y.astype(np.float32)).view(np.ndarray)
whisper_audio.load_audio = _safe_load

# =============================================================================
# Streamlit configuration & UI title
# =============================================================================
st.set_page_config(page_title="🎙️ RAISE", layout="wide")
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
• Upload WAV/MP3 (<15MB) or record live  
• Pipeline runs automatically  
• View each stage & interact  
• Download ZIP of all outputs
""")
    if st.button("Reset App"):
        st.session_state.clear()
        st.rerun()
    st.divider()
    st.markdown("**Keep API keys secure via .env**")

input_method = st.sidebar.radio("Input Method", ["Upload File", "Live Recording"])

# =============================================================================
# Load environment & initialize OpenAI
# =============================================================================
load_dotenv()
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("❌ Please set OPENAI_API_KEY in your environment.")
    st.stop()
openai.api_key = OPENAI_API_KEY

# =============================================================================
# Cached model loaders
# =============================================================================
use_cuda         = torch.cuda.is_available()
whisper_device   = "cuda" if use_cuda else "cpu"
transformers_dev = "cuda" if use_cuda else "cpu"

@st.cache_resource
def load_whisper_model():
    m = whisper.load_model("base", device=whisper_device)
    m.to(whisper_device)
    if whisper_device == "cpu":
        m.half = lambda: m
    return m


@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1,
        framework="pt",
        device=0 if use_cuda else -1
    )

@st.cache_resource
def load_diar_model(token):
    return DiarizationPipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=token
    )

@st.cache_resource
def load_speaker_embedding_model():
    return PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb",
        device=transformers_dev
    )

# Preload models in background threads
threading.Thread(target=load_whisper_model, daemon=True).start()
threading.Thread(target=load_emotion_model, daemon=True).start()
if HUGGINGFACE_TOKEN:
    threading.Thread(target=lambda: load_diar_model(HUGGINGFACE_TOKEN), daemon=True).start()
    threading.Thread(target=load_speaker_embedding_model, daemon=True).start()

# =============================================================================
# Sidebar: Model Status Indicators
# =============================================================================
with st.sidebar:
    st.header("Model Status")
    status_items = [
        ("Whisper", load_whisper_model),
        ("Emotion", load_emotion_model),
        ("Diarization", lambda: load_diar_model(HUGGINGFACE_TOKEN)),
        ("Speaker Embedding", load_speaker_embedding_model),
    ]
    for name, loader in status_items:
        try:
            loader()
            st.success(f"✅ {name} loaded")
        except Exception:
            st.error(f"❌ {name} failed")

# =============================================================================
# AudioRecorder for live input: normalize + low-pass
# =============================================================================
class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.frames = []
    def recv(self, frame):
        arr = np.array(frame.to_ndarray(), copy=True).astype(np.float32) / 32768.0
        arr = normalize_audio_array(arr)
        arr = lowpass_filter(arr, sr=44100)
        self.frames.append(arr)
        return frame

def save_wav(frames, path, sr=44100):
    audio = np.concatenate(frames, axis=0)
    audio = normalize_audio_array(audio)
    audio = lowpass_filter(audio, sr)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

# =============================================================================
# Preprocessing helpers for diarization
# =============================================================================
def verify_audio_format(path: str):
    info = sf.info(path)
    if info.samplerate != 16000 or info.channels != 1:
        raise ValueError("Audio must be mono WAV at 16kHz sample rate.")

def validate_audio_duration(path: str):
    y, sr = librosa.load(path, sr=16000, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    if duration < 2.0:
        raise ValueError("Audio too short (<2s) for reliable diarization.")
    return duration

def preprocess_for_diar(path: str):
    y, sr0 = sf.read(path)
    y_mono = y.mean(axis=1) if y.ndim > 1 else y
    if sr0 != 16000:
        y_mono = librosa.resample(y_mono, orig_sr=sr0, target_sr=16000)
    sf.write(path, y_mono, 16000, subtype="PCM_16")

# =============================================================================
# Core pipeline
# =============================================================================
def run_pipeline(orig_path: str):
    outdir    = tempfile.mkdtemp(prefix="raise_")
    work_path = os.path.join(outdir, "input.wav")
    shutil.copy2(orig_path, work_path)
    metrics = {}

    # Measure duration
    try:
        with wave.open(work_path, "rb") as wf:
            dur = wf.getnframes() / wf.getframerate()
    except:
        dur = None
    metrics["audio_length_sec"] = dur

    # Empty check
    data = whisper_audio.load_audio(work_path)
    if data is None or len(data) == 0:
        st.error("🚫 Audio empty or unreadable.")
        return None

    # --- Transcription ---
    with st.spinner("Transcribing…"):
        try:
            metrics["transcription_start"] = datetime.utcnow().isoformat()
            wh_model = load_whisper_model()
            res = wh_model.transcribe(work_path, word_timestamps=True)
            segs = res.get("segments", [])
            txt_raw = res.get("text", "").strip()

            txt = txt_raw

            # Save transcripts
            open(os.path.join(outdir, "raw_transcript.txt"), "w",
                 encoding="utf-8").write(txt_raw)
            open(os.path.join(outdir, "full_transcript.json"), "w").write(
                json.dumps({"raw": txt_raw, "segments": segs})
            )

            metrics["transcription_end"] = datetime.utcnow().isoformat()
            metrics["transcription_duration_sec"] = (
                datetime.fromisoformat(metrics["transcription_end"])
                - datetime.fromisoformat(metrics["transcription_start"])
            ).total_seconds()

            st.success("✅ Transcription done")
            st.subheader("Transcript")
            st.text_area("Transcript", txt, height=150)
        except Exception:
            logger.exception("Transcription failed")
            st.error("❌ Transcription error.")
            return None

    # --- Diarization + Clustering ---
    if dur is None or dur < 2.0 or not HUGGINGFACE_TOKEN:
        st.warning("⚠️ Skipping diarization (short or no token).")
        cleaned_segments = []
        speaker_map = {}
    else:
        with st.spinner("Running diarization…"):
            try:
                preprocess_for_diar(work_path)
                verify_audio_format(work_path)
                validate_audio_duration(work_path)

                metrics["diarization_start"] = datetime.utcnow().isoformat()
                diar_model = load_diar_model(HUGGINGFACE_TOKEN)
                diar = diar_model(work_path)
                metrics["diarization_end"] = datetime.utcnow().isoformat()
                metrics["diarization_duration_sec"] = (
                    datetime.fromisoformat(metrics["diarization_end"])
                    - datetime.fromisoformat(metrics["diarization_start"])
                ).total_seconds()
                st.success("✅ Diarization done")

                # Raw segments
                raw_segs = [
                    {"start": seg.start, "end": seg.end, "label": lbl}
                    for seg, _, lbl in diar.itertracks(yield_label=True)
                ]
                # Merge small gaps & filter short
                merged = []
                for s in sorted(raw_segs, key=lambda x: x["start"]):
                    if (merged and s["label"] == merged[-1]["label"]
                            and s["start"] - merged[-1]["end"] < 0.8):
                        merged[-1]["end"] = max(merged[-1]["end"], s["end"])
                    else:
                        merged.append(s)
                cleaned_segments = [s for s in merged if s["end"] - s["start"] >= 1.0]

                # Load waveform & resample once
                waveform, sr_wav = torchaudio.load(work_path)
                if sr_wav != 16000:
                    waveform = Resample(sr_wav, 16000)(waveform)

                # Chunked embedding extraction
                emb_model = load_speaker_embedding_model()
                embs = []
                batch_size = 8
                for i in range(0, len(cleaned_segments), batch_size):
                    batch = cleaned_segments[i:i+batch_size]
                    wavs = []
                    max_len = 0
                    for cs in batch:
                        start = int(cs["start"] * 16000)
                        end = int(cs["end"] * 16000)
                        wav_seg = waveform[:, start:end]
                        wavs.append(wav_seg)
                        max_len = max(max_len, wav_seg.shape[1])

                    # Pad all wavs to max_len
                    padded_wavs = []
                    for wav_seg in wavs:
                        if wav_seg.shape[1] < max_len:
                            pad_size = max_len - wav_seg.shape[1]
                            pad = torch.zeros((wav_seg.shape[0], pad_size), device=wav_seg.device)
                            wav_seg = torch.cat([wav_seg, pad], dim=1)
                        padded_wavs.append(wav_seg)

                    batch_tensor = torch.stack(padded_wavs, dim=0).to(transformers_dev)
                    out_embs = emb_model(batch_tensor)
                    for emb in out_embs:
                        embs.append(emb.flatten())
                    del batch_tensor, out_embs
                    torch.cuda.empty_cache()

                embs = np.stack(embs) if embs else np.empty((0,0))

                # Clustering
                if embs.shape[0] >= 2:
                    scores = {}
                    max_k = min(5, embs.shape[0])
                    for k in range(2, max_k+1):
                        labs = AgglomerativeClustering(n_clusters=k).fit_predict(embs)
                        scores[k] = silhouette_score(embs, labs)
                    best_k = max(scores, key=scores.get)
                    clusterer = AgglomerativeClustering(n_clusters=best_k)
                    labs = clusterer.fit_predict(embs)
                    for idx, cs in enumerate(cleaned_segments):
                        cs["cluster"] = int(labs[idx])
                    uniq = sorted(set(labs))
                    speaker_map = {c: f"Speaker {i+1}" for i,c in enumerate(uniq)}
                else:
                    seen = []
                    for cs in cleaned_segments:
                        if cs["label"] not in seen:
                            seen.append(cs["label"])
                    speaker_map = {lbl: f"Speaker {i+1}" for i,lbl in enumerate(seen)}

                st.subheader("Speaker Segments")
                st.json([
                    {
                        "start": s["start"],
                        "end": s["end"],
                        "speaker": speaker_map.get(s.get("cluster", s["label"]), s["label"])
                    }
                    for s in cleaned_segments
                ])
            except Exception:
                logger.exception("Diarization/Clustering failed")
                st.error("❌ Diarization error.")
                cleaned_segments = []
                speaker_map = {}

    # --- Emotion Analysis ---
    with st.spinner("Analyzing emotions…"):
        try:
            metrics["emotion_start"] = datetime.utcnow().isoformat()
            emo_model = load_emotion_model()
            texts = [seg["text"].strip() for seg in segs]
            emos = emo_model(texts, batch_size=8)

            annotated = []
            for i, seg in enumerate(segs):
                assigned = "Speaker 1"
                overlaps = [
                    cs for cs in cleaned_segments
                    if cs["start"] < seg["end"] and cs["end"] > seg["start"]
                ]
                if overlaps:
                    c = overlaps[0].get("cluster")
                    assigned = speaker_map.get(c, assigned)
                lbl = emos[i][0].get("label", "neutral").lower() if texts[i] else "neutral"
                annotated.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "speaker": assigned,
                    "emotion": lbl
                })

            open(os.path.join(outdir, "annotated_output.json"), "w").write(
                json.dumps(annotated, indent=2)
            )

            metrics["emotion_end"] = datetime.utcnow().isoformat()
            metrics["emotion_duration_sec"] = (
                datetime.fromisoformat(metrics["emotion_end"])
                - datetime.fromisoformat(metrics["emotion_start"])
            ).total_seconds()

            st.success("✅ Emotion done")
            st.subheader("Annotated Transcript")
            for e in annotated:
                s0 = str(timedelta(seconds=int(e["start"])))
                s1 = str(timedelta(seconds=int(e["end"])))
                st.markdown(f"**[{s0}->{s1}] {e['speaker']}** {e['text']}")
                st.info(f"Emotion: {e['emotion']}")
        except Exception:
            logger.exception("Emotion analysis failed")
            st.error("❌ Emotion error.")
            annotated = []

    # --- Emotion Timeline Plot with Legend ---
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        fig, ax = plt.subplots(figsize=(10, 2))
        spks = sorted({a["speaker"] for a in annotated})
        idx_map = {s: i for i,s in enumerate(spks)}
        cols = {
            "joy": "green", "anger": "red", "sadness": "blue",
            "neutral": "gray", "surprise": "orange",
            "fear": "purple", "disgust": "brown"
        }
        for a in annotated:
            ax.barh(
                idx_map[a["speaker"]],
                a["end"] - a["start"],
                left=a["start"],
                color=cols.get(a["emotion"], "black"),
                edgecolor="black",
                height=0.4
            )
        ax.set_yticks(list(idx_map.values()))
        ax.set_yticklabels(list(idx_map.keys()))
        ax.set_xlabel("Time (s)")

        # Legend
        handles = [Patch(facecolor=cols[e], label=e) for e in cols]
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        graph_path = os.path.join(outdir, "emotion_graph.png")
        plt.savefig(graph_path)
        st.image(graph_path, caption="Emotion Timeline")
    except Exception:
        logger.exception("Timeline plot failed")

    # --- GPT Feedback ---
    with st.spinner("Generating GPT feedback…"):
        try:
            metrics["gpt_start"] = datetime.utcnow().isoformat()
            prompt = "You are a conversation coach. Provide feedback:\n\n" + txt
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            feedback = resp.choices[0].message.content
            st.text_area("💬 GPT Feedback", feedback, height=200)
            metrics["gpt_end"] = datetime.utcnow().isoformat()
            metrics["gpt_duration_sec"] = (
                datetime.fromisoformat(metrics["gpt_end"])
                - datetime.fromisoformat(metrics["gpt_start"])
            ).total_seconds()
            st.success("✅ GPT feedback done")
        except openai.error.RateLimitError:
            st.error("⚠️ GPT rate limit exceeded.")
            feedback = "[Rate limit exceeded]"
        except Exception:
            logger.exception("GPT feedback failed")
            st.error("❌ GPT error.")
            feedback = "[Feedback unavailable]"
        open(os.path.join(outdir, "gpt_feedback.txt"), "w").write(feedback)

    # --- Total duration & metrics save ---
    try:
        total = datetime.utcnow() - datetime.fromisoformat(metrics["transcription_start"])
        metrics["total_duration_sec"] = total.total_seconds()
    except:
        pass
    open(os.path.join(outdir, "latency_metrics.json"), "w").write(
        json.dumps(metrics, indent=2)
    )

    # --- ZIP Archive ---
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    zfname = f"raise_{ts}_{uuid.uuid4().hex[:6]}.zip"
    zpath = os.path.join(tempfile.gettempdir(), zfname)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in os.listdir(outdir):
            zf.write(os.path.join(outdir, fn), fn)

    return {
        "zip":    zpath,
        "full_text": txt,
        "annotated": annotated,
        "feedback":  feedback,
        "metrics":   metrics
    }

# =============================================================================
# Main UI logic
# =============================================================================
def main():
    st.title("🎙️ RAISE – Zero-Click + Interactive UI + Metrics")
    res = None

    if input_method == "Upload File":
        up = st.file_uploader("Upload WAV/MP3 (<15MB)", type=["wav", "mp3"])
        if up:
            if up.size > 15 * 1024 * 1024:
                st.warning("⚠️ File too large. Please use <15MB.")
            else:
                tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                if up.type == "audio/mpeg" or up.name.endswith(".mp3"):
                    mp3f = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    mp3f.write(up.read()); mp3f.close()
                    audio = AudioSegment.from_mp3(mp3f.name)
                    audio.set_channels(1).set_frame_rate(16000).export(
                        tmpf.name, format="wav"
                    )
                    os.remove(mp3f.name)
                else:
                    tmpf.write(up.read()); tmpf.close()
                res = run_pipeline(tmpf.name)
    else:
        st.info("Recording… click Stop to analyze.")
        ctx = webrtc_streamer(
            key="live",
            mode=WebRtcMode.SENDONLY,
            media_stream_constraints={"audio": True},
            rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
            audio_processor_factory=AudioRecorder
        )
        if st.button("Stop & Analyze"):
            if ctx.audio_processor and ctx.audio_processor.frames:
                tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                save_wav(ctx.audio_processor.frames, tmpf.name)
                res = run_pipeline(tmpf.name)
            else:
                st.error("🚫 No audio captured.")

    if res:
        # Q&A chat
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        q = st.text_input("Ask a question")
        if q:
            try:
                ans = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role":"user","content":
                               f"Transcript:\n{res['full_text']}\nQuestion:{q}"}],
                    temperature=0.7
                ).choices[0].message.content
                st.session_state.chat_history.append((q, ans))
            except Exception:
                st.error("❌ Q&A failed.")
        for u, a in reversed(st.session_state.chat_history):
            st.markdown(f"**You:** {u}")
            st.markdown(f"**Bot:** {a}")

        # Latency metrics
        with st.expander("📊 Latency Metrics"):
            for k, v in res["metrics"].items():
                if k.endswith("_sec"):
                    st.write(f"**{k.replace('_',' ').capitalize()}**: {v:.2f} sec")

        # Emotion summary
        st.subheader("Emotion Summary")
        st.bar_chart(dict(Counter(e["emotion"] for e in res["annotated"])))

        # Download ZIP
        if st.checkbox("Download ZIP", True):
            with open(res["zip"], "rb") as f:
                st.download_button(
                    "📦 Download ZIP",
                    data=f,
                    file_name=os.path.basename(res["zip"]),
                    mime="application/zip"
                )

if __name__ == "__main__":
    main()
