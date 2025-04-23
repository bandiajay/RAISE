# =============================================================================
# EchoSense – Conversational Analysis App (app.py)
# =============================================================================
# Phase 1: Full Automation + Metrics + Export + Interactive UI + Enhancements
# =============================================================================

import sys
# Silence torch.classes warnings
sys.modules["torch.classes"] = None

import os
import warnings
import json
import wave
import shutil
import zipfile
import logging
import tempfile
import uuid
import numpy as np
import soundfile as sf
from datetime import datetime, timedelta
from collections import Counter

# =============================================================================
# Monkey-patch to bypass permission errors copying hyperparams in speechbrain
# =============================================================================
_orig_shutil_copy = shutil.copy
def _safe_copy(src, dst):
    try:
        return _orig_shutil_copy(src, dst)
    except (PermissionError, OSError) as e:
        warnings.warn(f"[EchoSense Patch] Skipped copying file due to permission: {e}")
shutil.copy = _safe_copy

# Patch os.symlink if used
import os as _os
_orig_symlink = _os.symlink
def _safe_symlink(src, dst):
    try:
        return _orig_symlink(src, dst)
    except (PermissionError, OSError) as e:
        warnings.warn(f"[EchoSense Patch] Skipped symlink due to: {e}")
_os.symlink = _safe_symlink

# =============================================================================
# Ensure np.NaN alias for NumPy 2.x
# =============================================================================
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import torch
import librosa
import whisper
import whisper.audio as whisper_audio
import whisper.model

from dotenv import load_dotenv
import streamlit as st
from transformers import pipeline
from pyannote.audio import Pipeline as DiarizationPipeline
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
from openai import OpenAI, RateLimitError

# =============================================================================
# Load environment & initialize OpenAI client
# =============================================================================
load_dotenv()
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("❌ Please set OPENAI_API_KEY in .env or environment.")
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

# =============================================================================
# Configure logging
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logger.info(f"NumPy {np.__version__}, Torch {torch.__version__}, Whisper {whisper.__version__}")

# =============================================================================
# GPU / Device setup
# =============================================================================
use_cuda = torch.cuda.is_available()
whisper_device = "cuda" if use_cuda else "cpu"
transformers_device = 0 if use_cuda else -1

# =============================================================================
# Patches for Whisper to handle NumPy subclass issues
# =============================================================================
_orig_from_numpy = torch.from_numpy
torch.from_numpy = lambda x: _orig_from_numpy(
    np.ascontiguousarray(x.astype(np.float32)).view(np.ndarray)
)
whisper.model.Whisper.set_alignment_heads = lambda self, alignment_heads=None: logger.warning("Skipping alignment patch")
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
# Preprocessing helpers for diarization
# =============================================================================
def verify_audio_format(path: str):
    info = sf.info(path)
    logger.info(f"Audio format: {info}")
    if info.samplerate != 16000 or info.channels != 1:
        raise ValueError("Audio must be mono WAV at 16kHz sample rate.")

def validate_audio_duration(path: str):
    y, sr = librosa.load(path, sr=16000, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    logger.info(f"Audio duration: {duration:.2f}s")
    if duration < 2.0:
        raise ValueError("Audio too short (<2s) for reliable diarization.")

def preprocess_for_diar(path: str):
    y, sr0 = sf.read(path)
    y_mono = y.mean(axis=1) if y.ndim > 1 else y
    if sr0 != 16000 or y.ndim > 1:
        y_res = librosa.resample(y_mono, orig_sr=sr0, target_sr=16000)
        sf.write(path, y_res, 16000)
        logger.info("Resampled audio to mono 16kHz WAV for diarization")

# =============================================================================
# Streamlit configuration
# =============================================================================
st.set_page_config(page_title="🎙️ EchoSense", layout="wide")
with st.sidebar:
    st.header("Instructions")
    st.markdown(
        """
• Upload WAV (<15MB) or record live  
• Pipeline runs automatically  
• View each stage & interact  
• Download ZIP of all outputs
        """
    )
    if st.button("Reset App"):
        st.session_state.clear()
        st.experimental_rerun()
    st.divider()
    st.markdown("**Keep API keys secure via .env**")
input_method = st.sidebar.radio("Input Method", ["Upload File", "Live Recording"])

# =============================================================================
# Cached model loaders
# =============================================================================
@st.cache_resource
def load_whisper():
    return whisper.load_model("base", device=whisper_device)

@st.cache_resource
def load_emotion():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1, framework="pt", device=transformers_device
    )

@st.cache_resource
def load_diar(token):
    return DiarizationPipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1", use_auth_token=token
    )

# =============================================================================
# Audio recorder for live input
# =============================================================================
class AudioRecorder(AudioProcessorBase):
    def __init__(self): self.frames = []
    def recv(self, frame):
        self.frames.append(np.array(frame.to_ndarray(), copy=True))
        return frame

def save_wav(frames, path, sr=44100):
    audio = np.concatenate(frames, axis=0)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

# =============================================================================
# Core processing pipeline
# =============================================================================
def run_pipeline(audio_path: str):
    # Measure audio length
    try:
        with wave.open(audio_path,'rb') as wf:
            dur = wf.getnframes() / wf.getframerate()
    except:
        dur = None
    outdir = tempfile.mkdtemp(prefix="echo_")
    metrics = {'audio_length_sec': dur}

    # Pre-check for empty audio
    data = whisper_audio.load_audio(audio_path)
    if data is None or len(data) == 0:
        st.error("🚫 Audio empty or unreadable.")
        return None

    try:
        # --- Transcription ---
        metrics['transcription_start'] = datetime.utcnow().isoformat()
        mdl = load_whisper()
        res = mdl.transcribe(audio_path, word_timestamps=True)
        segs = res.get('segments', []) if isinstance(res, dict) else [{'start':0,'end':0,'text':res.text}]
        txt = (res.get('text','') if isinstance(res, dict) else res.text).strip()
        open(os.path.join(outdir,'full_transcript.txt'),'w',encoding='utf-8').write(txt)
        open(os.path.join(outdir,'full_transcript.json'),'w',encoding='utf-8').write(json.dumps({'text':txt,'segments':segs}))
        metrics['transcription_end'] = datetime.utcnow().isoformat()
        metrics['transcription_duration_sec'] = (
            datetime.fromisoformat(metrics['transcription_end']) -
            datetime.fromisoformat(metrics['transcription_start'])
        ).total_seconds()
        st.success("✅ Transcription done")

        # --- Diarization ---
        if HUGGINGFACE_TOKEN:
            try:
                preprocess_for_diar(audio_path)
                verify_audio_format(audio_path)
                validate_audio_duration(audio_path)
                metrics['diarization_start'] = datetime.utcnow().isoformat()
                st.info("🔍 Running speaker diarization…")
                pipeline = load_diar(HUGGINGFACE_TOKEN)
                dia = pipeline(audio_path)
                metrics['diarization_end'] = datetime.utcnow().isoformat()
                metrics['diarization_duration_sec'] = (
                    datetime.fromisoformat(metrics['diarization_end']) -
                    datetime.fromisoformat(metrics['diarization_start'])
                ).total_seconds()
                st.success("✅ Diarization done")
            except Exception as e:
                dia = None
                st.error(f"⚠️ Diarization failed: {e}")
        else:
            dia = None
            st.warning("⚠️ Skipped diarization")

        # --- Emotion Analysis ---
        metrics['emotion_start'] = datetime.utcnow().isoformat()
        emo = load_emotion()
        texts = [s['text'].strip() for s in segs]
        eres = emo(texts, batch_size=8)
        ann = []
        for i, s in enumerate(segs):
            spk = f"Speaker {i%2+1}"
            if dia:
                for t, _, sp in dia.itertracks(yield_label=True):
                    if t.start <= s['start'] <= t.end:
                        spk = sp
                        break
            lbl = eres[i][0].get('label','neutral').lower() if texts[i] else 'neutral'
            ann.append({**s, 'speaker': spk, 'emotion': lbl})
        open(os.path.join(outdir,'annotated_output.json'),'w').write(json.dumps(ann, indent=2))
        metrics['emotion_end'] = datetime.utcnow().isoformat()
        metrics['emotion_duration_sec'] = (
            datetime.fromisoformat(metrics['emotion_end']) -
            datetime.fromisoformat(metrics['emotion_start'])
        ).total_seconds()
        st.success("✅ Emotion done")

        # --- Emotion Timeline Graph ---
        figp = os.path.join(outdir, 'emotion_graph.png')
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch
            spks = sorted({a['speaker'] for a in ann})
            mp = {s: i for i, s in enumerate(spks)}
            cols = {'joy': 'green', 'anger': 'red', 'sadness': 'blue', 'neutral': 'gray', 'surprise': 'orange', 'fear': 'purple', 'disgust': 'brown'}
            fig, ax = plt.subplots(figsize=(10, 2))
            for a in ann:
                ax.barh(mp[a['speaker']], a['end'] - a['start'], left=a['start'], color=cols.get(a['emotion'], 'black'), edgecolor='black', height=0.4)
            ax.set_yticks(list(mp.values()))
            ax.set_yticklabels(list(mp.keys()))
            ax.set_xlabel('Time (s)')
            plt.tight_layout()
            plt.savefig(figp)
            st.image(figp, caption='Emotion Timeline')
        except Exception as e:
            logger.error(f"Emotion graph error: {e}")

        # --- GPT Feedback ---
        metrics['gpt_start'] = datetime.utcnow().isoformat()
        try:
            prompt = "You are a conversation coach. Provide feedback:\n\n" + txt
            rr = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{'role': 'user', 'content': prompt}], temperature=0.7)
            fb = rr.choices[0].message.content
            st.text_area("💬 GPT Feedback", fb, height=200)
            st.success("✅ GPT feedback done")
        except RateLimitError as e:
            fb = "[Rate limit exceeded]"
            st.error("⚠️ GPT rate limit exceeded.")
            logger.error(f"GPT rate limit: {e}")
        except Exception as e:
            fb = "[Feedback unavailable]"
            st.error("⚠️ GPT failed.")
            logger.error(f"GPT error: {e}")
        open(os.path.join(outdir, 'gpt_feedback.txt'), 'w').write(fb)
        metrics['gpt_end'] = datetime.utcnow().isoformat()
        metrics['gpt_duration_sec'] = (
            datetime.fromisoformat(metrics['gpt_end']) - datetime.fromisoformat(metrics['gpt_start'])
        ).total_seconds()

        # --- Total Duration ---
        metrics['total_duration_sec'] = (
            datetime.fromisoformat(metrics['gpt_end']) - datetime.fromisoformat(metrics['transcription_start'])
        ).total_seconds()

        # --- Save Metrics ---
        open(os.path.join(outdir, 'latency_metrics.json'), 'w').write(json.dumps(metrics, indent=2))

        # --- Create ZIP ---
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        zf = f"echo_{ts}_{uuid.uuid4().hex[:6]}.zip"
        zp = os.path.join(tempfile.gettempdir(), zf)
        with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as zfobj:
            for fn in os.listdir(outdir):
                zfobj.write(os.path.join(outdir, fn), fn)

        return {'zip': zp, 'full_text': txt, 'annotated': ann, 'feedback': fb, 'metrics': metrics}

    finally:
        if os.path.exists(audio_path): os.remove(audio_path)
        shutil.rmtree(outdir, ignore_errors=True)

# =============================================================================
# Main application logic and UI
# =============================================================================
def main():
    st.title("🎙️ EchoSense – Zero-Click + Interactive UI + Metrics")
    res = None

    if input_method == 'Upload File':
        up = st.file_uploader("Upload WAV (<15MB)", type=['wav'])
        if up:
            if up.size > 15*1024*1024:
                st.warning("⚠️ File too large. Please use <15MB.")
            else:
                tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                tmpf.write(up.read()); tmpf.close()
                res = run_pipeline(tmpf.name)
    else:
        st.info("Recording… click Stop to analyze.")
        ctx = webrtc_streamer(
            key='live',
            mode=WebRtcMode.SENDONLY,
            media_stream_constraints={'audio': True},
            rtc_configuration={'iceServers': [{'urls': ['stun:stun.l.google.com:19302']}]},
            audio_processor_factory=AudioRecorder
        )
        if st.button("Stop & Analyze"):
            if ctx.audio_processor and ctx.audio_processor.frames:
                tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                save_wav(ctx.audio_processor.frames, tmpf.name)
                res = run_pipeline(tmpf.name)
            else:
                st.error("No audio captured.")

    if res:
        st.subheader("Transcript")
        st.text_area("Transcript", res['full_text'], height=200)
        
        st.subheader("Annotated Transcript")
        for e in res['annotated']:
            s = str(timedelta(seconds=int(e['start'])))
            e2 = str(timedelta(seconds=int(e['end'])))
            st.markdown(f"**[{s}->{e2}] {e['speaker']}** {e['text']}")
            st.info(f"Emotion: {e['emotion']}")
        
        st.subheader("GPT Feedback")
        st.text_area("Feedback", res['feedback'], height=200)

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        q = st.text_input("Ask a question")
        if q:
            try:
                qa = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{'role':'user','content':f"Transcript:\n{res['full_text']}\nQuestion: {q}"}],
                    temperature=0.7
                )
                a = qa.choices[0].message.content
                st.session_state.chat_history.append((q, a))
            except Exception as e:
                st.error(f"Q&A failed: {e}")
        for u, a in reversed(st.session_state.chat_history):
            st.markdown(f"**You:** {u}")
            st.markdown(f"**Bot:** {a}")

        with st.expander("📊 Latency Metrics"):
            for k, v in res['metrics'].items():
                if k.endswith('_duration_sec') or k == 'audio_length_sec':
                    st.write(f"**{k.replace('_',' ').capitalize()}**: {v:.2f} sec")

        st.subheader("Emotion Summary")
        st.bar_chart(dict(Counter(e['emotion'] for e in res['annotated'])))

        if st.checkbox("Show download", True):
            with open(res['zip'], 'rb') as f:
                st.download_button(
                    "📦 Download ZIP",
                    data=f,
                    file_name=os.path.basename(res['zip']),
                    mime="application/zip"
                )

if __name__ == '__main__':
    main()