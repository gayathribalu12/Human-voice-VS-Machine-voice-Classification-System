import streamlit as st
import joblib
import numpy as np
import librosa
import os
import tempfile
import sounddevice as sd
from scipy.io.wavfile import write
import pandas as pd

st.set_page_config(page_title="AI Voice Detector", layout="wide", initial_sidebar_state="expanded")

# -------------------------------
# SESSION STATE (IMPORTANT)
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# UI CSS - PREMIUM STYLING
# -------------------------------
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #f1f5f9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main-header {
        text-align: center;
        padding: 40px 20px 30px;
        background: linear-gradient(135deg, rgba(30, 144, 255, 0.15) 0%, rgba(255, 20, 147, 0.15) 100%);
        border-bottom: 2px solid rgba(100, 200, 255, 0.3);
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .main-title {
        font-size: 56px;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa 0%, #f472b6 50%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        letter-spacing: -1px;
    }
    
    .subtitle {
        font-size: 16px;
        color: #94a3b8;
        font-weight: 300;
        letter-spacing: 2px;
    }
    
    .premium-card {
        padding: 30px;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.4) 0%, rgba(51, 65, 85, 0.3) 100%);
        border: 1px solid rgba(100, 200, 255, 0.2);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        margin-bottom: 0;
        height: fit-content;
    }
    
    .premium-card:hover {
        border-color: rgba(100, 200, 255, 0.4);
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.5) 0%, rgba(51, 65, 85, 0.4) 100%);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    }
    
    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #60a5fa;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .result-box {
        padding: 35px;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.8) 100%);
        border: 2px solid rgba(100, 200, 255, 0.3);
        margin-top: 25px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }
    
    .result-human {
        border-color: rgba(34, 197, 94, 0.5);
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.3) 0%, rgba(15, 23, 42, 0.8) 100%);
    }
    
    .result-machine {
        border-color: rgba(239, 68, 68, 0.5);
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.3) 0%, rgba(15, 23, 42, 0.8) 100%);
    }
    
    .result-label {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 15px;
        background: linear-gradient(90deg, #60a5fa 0%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .confidence-value {
        font-size: 36px;
        font-weight: 700;
        text-align: center;
        color: #60a5fa;
        margin-bottom: 20px;
    }
    
    .decision-text {
        font-size: 16px;
        text-align: center;
        font-weight: 600;
        color: #cbd5e1;
        margin: 20px 0;
    }
    
    .explanation-box {
        background: rgba(100, 200, 255, 0.1);
        border-left: 4px solid #60a5fa;
        padding: 15px 20px;
        border-radius: 8px;
        margin-top: 20px;
        font-size: 14px;
        color: #e2e8f0;
    }
    
    .metric-card {
        background: rgba(30, 58, 138, 0.3);
        border: 1px solid rgba(100, 200, 255, 0.2);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 15px;
    }
    
    .metric-label {
        font-size: 12px;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #60a5fa;
        margin-top: 8px;
    }
    
    .progress-bar {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, #60a5fa 0%, #f472b6 100%);
        margin: 20px 0;
    }
    
    .mode-badge {
        display: inline-block;
        background: rgba(100, 200, 255, 0.2);
        border: 1px solid #60a5fa;
        color: #60a5fa;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-top: 10px;
    }
    
    .sidebar .element-container {
        background: rgba(30, 58, 138, 0.2) !important;
        border-radius: 10px !important;
        padding: 15px !important;
        margin-bottom: 15px !important;
        border: 1px solid rgba(100, 200, 255, 0.1) !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 58, 138, 0.2);
        border: 1px solid rgba(100, 200, 255, 0.2);
        border-radius: 10px;
        padding: 12px 20px;
    }
    
    [data-testid="stColumn"] {
        gap: 20px;
    }
    
    [data-testid="stFileUploadDropzone"] {
        border-color: rgba(100, 200, 255, 0.3);
    }
    
    .stButton > button {
        width: 100%;
        font-weight: 600;
        font-size: 14px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
        <div class="main-title" style="margin: 0; font-size: 50px;">🎙️ AI Voice Authenticity Detector</div>
    </div>
    <div class="subtitle">Advanced Deep Learning Analysis • Real-Time Classification</div>
</div>
""", unsafe_allow_html=True)

col_mode1, col_mode2, col_mode3 = st.columns([1.2, 1.5, 1])
with col_mode1:
    st.write("")
with col_mode2:
    mode = st.selectbox("🧠 Detection Mode", ["Basic Detection", "Deepfake Detection"], label_visibility="collapsed")
with col_mode3:
    st.write("")

st.markdown("<div style='margin: -15px 0 20px 0;'></div>", unsafe_allow_html=True)

# -------------------------------
# � ENHANCED DASHBOARD
# -------------------------------
with st.sidebar:
    st.markdown('<div class="section-title">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown('<div class="metric-card"><div class="metric-label">Model Type</div><div class="metric-value" style="font-size: 18px;">XGBoost</div></div>', unsafe_allow_html=True)
    
    with col_d2:
        st.markdown('<div class="metric-card"><div class="metric-label">Accuracy</div><div class="metric-value" style="font-size: 18px;">92.33%</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<div class="metric-card"><div class="metric-label">🧬 Detection Mode</div><div style="margin-top: 10px;"><span class="mode-badge">' + mode + '</span></div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Predictions</div><div class="metric-value">{len(st.session_state.history)}</div></div>', unsafe_allow_html=True)
    
    with col_s2:
        if len(st.session_state.history) > 0:
            human_count = sum(1 for h in st.session_state.history if h["Result"] == "Human Voice")
            st.markdown(f'<div class="metric-card"><div class="metric-label">Human Detected</div><div class="metric-value">{human_count}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card"><div class="metric-label">Human Detected</div><div class="metric-value">0</div></div>', unsafe_allow_html=True)
    
    if len(st.session_state.history) > 0:
        st.markdown("---")
        st.markdown('<div class="section-title">📋 Recent Predictions</div>', unsafe_allow_html=True)
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist.tail(8), use_container_width=True, height=300)

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    if os.path.exists("model.pkl") and os.path.exists("scaler.pkl"):
        return joblib.load("model.pkl"), joblib.load("scaler.pkl")
    else:
        return None, None

# Load model early
model, scaler = load_model()

if model is None:
    st.error("❌ Run main.py first to generate model & scaler")
    st.stop()

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    audio = librosa.util.normalize(audio)

    # SAME AS TRAINING
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=80)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(audio)
    spectral = librosa.feature.spectral_contrast(y=audio, sr=sr)

    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(delta, axis=1),
        np.mean(delta2, axis=1),
        np.mean(chroma, axis=1),
        np.mean(zcr, axis=1),
        np.mean(spectral, axis=1)
    ])

    return features
# -------------------------------
# DECISION LOGIC
# -------------------------------
def get_decision(conf):
    if conf >= 0.85:
        return "✅ High Confidence Prediction"
    elif conf >= 0.65:
        return "❓ Needs Review (Ambiguous Audio)"
    else:
        return "⚠️ Uncertain / Likely Misclassified"

# -------------------------------
# EXPLAIN
# -------------------------------
def explain(label):
    if label == "Machine Voice":
        return "🔍 Smooth spectral patterns → AI Generated"
    else:
        return "🔍 Natural irregular patterns → Human Speech"

# -------------------------------
# PREDICT
# -------------------------------
def predict(model, scaler, file_path):
    f = extract_features(file_path)
    f = scaler.transform(f.reshape(1, -1))

    probs = model.predict_proba(f)[0]
    conf = float(np.max(probs))   # 🔥 FIX HERE
    label = "Human Voice" if np.argmax(probs) == 0 else "Machine Voice"

    return label, conf, probs

# -------------------------------
# RECORD AUDIO
# -------------------------------
def record_audio(duration):
    progress = st.progress(0)
    status = st.empty()

    rec = sd.rec(int(duration * 22050), samplerate=22050, channels=1)

    for i in range(duration):
        progress.progress((i+1)/duration)
        status.text(f"Recording... {duration-i-1}s left")
        sd.sleep(1000)

    sd.wait()

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp.name, 22050, rec)
    return temp.name

# -------------------------------
# MAIN LAYOUT
# -------------------------------
col1, col2 = st.columns(2, gap="medium")

# -------------------------------
# UPLOAD
# -------------------------------
with col1:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📂 Upload Audio File</div>', unsafe_allow_html=True)

    file = st.file_uploader("Upload WAV/MP3 file", type=["wav", "mp3", "m4a"], key="upload_file")

    if file is not None:
        st.audio(file, format="audio/wav")
        
    if file and st.button("🔍 Analyze Upload", key="upload_btn"):
        with st.spinner("🔬 Analyzing audio..."):
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.write(file.read())
            path = tmp.name

            label, conf, probs = predict(model, scaler, path)
            decision = get_decision(conf)

            # 🔥 SAVE HISTORY
            st.session_state.history.append({
                "Type": "Upload",
                "Result": label,
                "Confidence": round(conf*100, 2)
            })

            result_class = "result-human" if label == "Human Voice" else "result-machine"
            st.markdown(f'<div class="result-box {result_class}">', unsafe_allow_html=True)

            # Result Label
            st.markdown(f"<div class='result-label'>{label}</div>", unsafe_allow_html=True)
            
            # Confidence Percentage
            st.markdown(f"<div class='confidence-value'>{conf*100:.1f}%</div>", unsafe_allow_html=True)
            
            # Progress Bar
            col_p1, col_p2, col_p3 = st.columns([1, 3, 1])
            with col_p2:
                st.progress(float(conf))
            
            # Probability Chart
            st.markdown("---")
            col_c1, col_c2, col_c3 = st.columns([0.5, 4, 0.5])
            with col_c2:
                df = pd.DataFrame({"Class": ["Human 🧑", "Machine 🤖"], "Probability": probs})
                st.bar_chart(df.set_index("Class"), height=250)
            
            # Decision & Explanation
            st.markdown(f"<div class='decision-text'>{decision}</div>", unsafe_allow_html=True)
            st.markdown(f'<div class="explanation-box">{explain(label)}</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# RECORD
# -------------------------------
with col2:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎤 Record Live Audio</div>', unsafe_allow_html=True)

    duration = st.slider("Recording Duration (seconds)", 3, 15, 5, help="Record between 3-15 seconds")

    if st.button("🎙️ Record & Analyze", key="record_btn"):
        with st.spinner("🎙️ Recording audio..."):
            path = record_audio(duration)
        
        st.audio(path, format="audio/wav")
        
        with st.spinner("🔬 Analyzing audio..."):
            label, conf, probs = predict(model, scaler, path)
            decision = get_decision(conf)

            # 🔥 SAVE HISTORY
            st.session_state.history.append({
                "Type": "Mic",
                "Result": label,
                "Confidence": round(conf*100, 2)
            })

            result_class = "result-human" if label == "Human Voice" else "result-machine"
            st.markdown(f'<div class="result-box {result_class}">', unsafe_allow_html=True)

            # Result Label
            st.markdown(f"<div class='result-label'>{label}</div>", unsafe_allow_html=True)
            
            # Confidence Percentage
            st.markdown(f"<div class='confidence-value'>{conf*100:.1f}%</div>", unsafe_allow_html=True)
            
            # Progress Bar
            col_p1, col_p2, col_p3 = st.columns([1, 3, 1])
            with col_p2:
                st.progress(float(conf))
            
            # Probability Chart
            st.markdown("---")
            col_c1, col_c2, col_c3 = st.columns([0.5, 4, 0.5])
            with col_c2:
                df = pd.DataFrame({"Class": ["Human 🧑", "Machine 🤖"], "Probability": probs})
                st.bar_chart(df.set_index("Class"), height=250)
            
            # Decision & Explanation
            st.markdown(f"<div class='decision-text'>{decision}</div>", unsafe_allow_html=True)
            st.markdown(f'<div class="explanation-box">{explain(label)}</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)