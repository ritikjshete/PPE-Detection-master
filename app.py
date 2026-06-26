import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from ultralytics import YOLO
import threading
import os
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import queue

# Resolve project root regardless of where Streamlit is launched from
BASE_DIR = Path(__file__).parent.resolve()
from dotenv import load_dotenv
from PIL import Image
from collections import deque

load_dotenv(dotenv_path=BASE_DIR / ".env")

PHOTO_DIR = "Photo-Data"
os.makedirs(PHOTO_DIR, exist_ok=True)

st.set_page_config(
    page_title="PPE Safety Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #0d0f14; color: #e8eaf0; }

    .main-header {
        font-family: 'Space Mono', monospace;
        font-size: 2rem; font-weight: 700; color: #f0f4ff;
        letter-spacing: -0.5px; padding: 1.5rem 0 0.25rem;
        border-bottom: 2px solid #ff6b35; margin-bottom: 0.25rem;
    }
    .main-subheader {
        font-size: 0.85rem; color: #7b8090; letter-spacing: 0.12em;
        text-transform: uppercase; margin-bottom: 2rem;
        font-family: 'Space Mono', monospace;
    }

    [data-testid="stSidebar"] { background: #12151c !important; border-right: 1px solid #1e2230; }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label { color: #b0b6c8 !important; font-size: 0.82rem; }

    .sidebar-section {
        background: #1a1d27; border: 1px solid #252836;
        border-radius: 10px; padding: 1rem 1.1rem; margin: 0.75rem 0;
    }
    .sidebar-section-title {
        font-family: 'Space Mono', monospace; font-size: 0.7rem; font-weight: 700;
        letter-spacing: 0.15em; text-transform: uppercase; color: #ff6b35; margin-bottom: 0.75rem;
    }
    .status-pill {
        display: inline-block; padding: 2px 10px; border-radius: 20px;
        font-size: 0.72rem; font-weight: 600; font-family: 'Space Mono', monospace;
    }
    .status-ok  { background: #1a3a2a; color: #4caf83; border: 1px solid #2d6b4a; }
    .status-err { background: #3a1a1a; color: #e05555; border: 1px solid #6b2d2d; }

    .metrics-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
    .metric-card {
        flex: 1; min-width: 130px; background: #1a1d27; border: 1px solid #252836;
        border-radius: 12px; padding: 1.1rem 1rem 0.9rem;
        position: relative; overflow: hidden; transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #3a3f55; }
    .metric-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0;
        height: 3px; border-radius: 12px 12px 0 0;
    }
    .metric-card.orange::before { background: #ff6b35; }
    .metric-card.green::before  { background: #4caf83; }
    .metric-card.blue::before   { background: #5b8def; }
    .metric-card.red::before    { background: #e05555; }
    .metric-card.purple::before { background: #9b72e8; }
    .metric-icon  { font-size: 1.1rem; margin-bottom: 0.3rem; color: #7b8090; }
    .metric-value { font-family: 'Space Mono', monospace; font-size: 1.9rem; font-weight: 700; color: #f0f4ff; line-height: 1; }
    .metric-label { font-size: 0.72rem; color: #7b8090; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.3rem; }

    .alert-banner {
        display: flex; align-items: center; gap: 0.75rem;
        padding: 0.85rem 1.25rem; border-radius: 10px; margin: 0.75rem 0;
        font-size: 0.88rem; font-weight: 500; border: 1px solid;
    }
    .alert-danger  { background:#2a1212; border-color:#6b2d2d; color:#f09090; }
    .alert-success { background:#122a1a; border-color:#2d6b4a; color:#90f0b8; }
    .alert-warning { background:#2a2212; border-color:#6b5a2d; color:#f0d090; }

    .stTabs [data-baseweb="tab-list"] { background: #12151c; border-bottom: 1px solid #1e2230; gap: 0; }
    .stTabs [data-baseweb="tab"] {
        color: #7b8090; font-family: 'Space Mono', monospace; font-size: 0.78rem;
        letter-spacing: 0.05em; padding: 0.75rem 1.25rem; border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] { color: #ff6b35 !important; border-bottom-color: #ff6b35 !important; background: transparent !important; }

    .stButton > button {
        background: #ff6b35; color: #fff; border: none; border-radius: 8px;
        font-family: 'Space Mono', monospace; font-size: 0.78rem; font-weight: 700;
        letter-spacing: 0.05em; padding: 0.55rem 1.4rem; transition: background 0.2s, transform 0.1s;
    }
    .stButton > button:hover { background: #e55a25; transform: translateY(-1px); }

    .stTextInput > div > div > input,
    .stSelectbox > div > div { background: #1a1d27 !important; border: 1px solid #252836 !important; color: #e8eaf0 !important; border-radius: 8px !important; }

    .section-title {
        font-family: 'Space Mono', monospace; font-size: 0.75rem; font-weight: 700;
        letter-spacing: 0.15em; text-transform: uppercase; color: #7b8090;
        margin: 1.5rem 0 0.75rem; padding-bottom: 0.4rem; border-bottom: 1px solid #1e2230;
    }
</style>
""", unsafe_allow_html=True)

# ── SVG icons (no emoji) ──────────────────────────────────────────────────────
ICO_PEOPLE  = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="#5b8def"  viewBox="0 0 24 24"><path d="M16 11c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3zM8 11c1.66 0 3-1.34 3-3S9.66 5 8 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5C15 14.17 10.33 13 8 13zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z"/></svg>'
ICO_HARDHAT = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="#ff6b35" viewBox="0 0 24 24"><path d="M12 2C8 2 5 5.5 5 9v1H3v2h18v-2h-2V9c0-3.5-3-7-7-7zm0 2c2.76 0 5 2.69 5 5v1H7V9c0-2.31 2.24-5 5-5zM3 14v2c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2v-2H3z"/></svg>'
ICO_MASK    = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="#4caf83" viewBox="0 0 24 24"><path d="M5.5 11h13l1.5-4H4L5.5 11zm6.5 8c3.31 0 6-1.79 6-4v-1H6v1c0 2.21 2.69 4 6 4z"/></svg>'
ICO_VEST    = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="#9b72e8" viewBox="0 0 24 24"><path d="M16 2l-4 4-4-4-6 4v8h4v8h12v-8h4V6z"/></svg>'
ICO_CHECK   = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="#4caf83" viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/></svg>'
ICO_WARN    = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="#e05555" viewBox="0 0 24 24"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/></svg>'
ICO_CHART   = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="#4caf83" viewBox="0 0 24 24"><path d="M3 3v18h18V3H3zm16 16H5V5h14v14zM7 10h2v7H7v-7zm4-3h2v10h-2V7zm4 6h2v4h-2v-4z"/></svg>'
ICO_VIOL    = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="#e05555" viewBox="0 0 24 24"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/></svg>'
ICO_PEAK    = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="#5b8def" viewBox="0 0 24 24"><path d="M16 11c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3zM8 11c1.66 0 3-1.34 3-3S9.66 5 8 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5C15 14.17 10.33 13 8 13zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z"/></svg>'
ICO_FRAMES  = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="#ff6b35" viewBox="0 0 24 24"><path d="M18 4l2 4h-3l-2-4h-2l2 4h-3l-2-4H8l2 4H7L5 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V4h-4z"/></svg>'


# ── PPEDetectionSystem ────────────────────────────────────────────────────────
class PPEDetectionSystem:
    MAX_HISTORY = 500

    def __init__(self):
        self.model             = None
        self.detection_history = deque(maxlen=self.MAX_HISTORY)
        self.violation_log     = deque(maxlen=500)
        email_pw = os.getenv("EMAIL_PASSWORD")
        self.email_config = {
            'sender':   os.getenv("SENDER_EMAIL"),
            'receiver': os.getenv("RECEIVER_EMAIL"),
            'password': email_pw.strip().replace(" ", "") if email_pw else None,
        }
        self._email_last_sent: dict[str, datetime] = {}
        self.colors = [
            (255, 80,  0), (50, 220,120), (220, 50, 50), (220,180,  0),
            (180, 50,220), (80, 160,255), (128,  0,128), (100,180, 50),
            (0,  180,200), (160,160,160),
        ]

    def load_model(self, model_path="Model/ppe.pt"):
        try:
            # Resolve to absolute path anchored at the project root
            abs_path = Path(model_path)
            if not abs_path.is_absolute():
                abs_path = BASE_DIR / model_path

            if not abs_path.exists():
                st.error(
                    f"Model file not found: `{abs_path}`\n\n"
                    "Make sure `ppe.pt` is inside the `Model/` folder of the project."
                )
                return False

            self.model = YOLO(str(abs_path))
            return True
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return False

    def draw_text_with_background(self, frame, text, position,
                                  font_scale=0.4, color=(255,255,255),
                                  thickness=1, bg_color=(0,0,0),
                                  alpha=0.7, padding=5):
        font     = cv2.FONT_HERSHEY_SIMPLEX
        tw, th   = cv2.getTextSize(text, font, font_scale, thickness)[0]
        overlay  = frame.copy()
        x, y     = position
        cv2.rectangle(overlay,
                      (x - padding, y - th - padding),
                      (x + tw + padding, y + padding), bg_color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

    # ── Throttled email ───────────────────────────────────────────────────────
    def _can_send_email(self, violation_type: str, cooldown_seconds: int) -> bool:
        last = self._email_last_sent.get(violation_type)
        return last is None or (datetime.now() - last).total_seconds() >= cooldown_seconds

    def send_email_alert(self, image_path: str, violation_type: str,
                         confidence_score: float, cooldown_seconds: int):
        if not self._can_send_email(violation_type, cooldown_seconds):
            return False
        try:
            msg = MIMEMultipart()
            msg["From"]    = self.email_config['sender']
            msg["To"]      = self.email_config['receiver']
            msg["Subject"] = f"PPE Safety Alert: {violation_type}"
            msg.attach(MIMEText(
                f"SAFETY VIOLATION DETECTED\n\n"
                f"Type      : {violation_type}\n"
                f"Confidence: {confidence_score * 100:.2f}%\n"
                f"Time      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"See attached image.\n\n-- PPE Detection System", "plain"))
            if os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header("Content-Disposition",
                                    f"attachment; filename={os.path.basename(image_path)}")
                    msg.attach(part)
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(self.email_config['sender'], self.email_config['password'])
                server.sendmail(self.email_config['sender'],
                                self.email_config['receiver'], msg.as_string())
            self._email_last_sent[violation_type] = datetime.now()
            return True
        except Exception as e:
            print(f"[Email Error] {e}")
            return False

    # ── Detection ─────────────────────────────────────────────────────────────
    def detect_objects(self, frame, confidence_threshold=0.5):
        if self.model is None:
            return frame, {}
        results    = self.model(frame)
        detections = {
            'hardhat':0, 'mask':0, 'safety_vest':0,
            'person':0, 'no_hardhat':0, 'no_mask':0, 'violations':[],
        }
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < confidence_threshold:
                    continue
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cls          = int(box.cls[0])
                name         = self.model.names[cls]
                color        = self.colors[cls % len(self.colors)]
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                self.draw_text_with_background(
                    frame, f"{name} {conf:.2f}", (x1, y1-10),
                    font_scale=0.4, color=(255,255,255), bg_color=color, alpha=0.8, padding=4)
                cn = name.lower()
                if   'hardhat' in cn and 'no' not in cn:  detections['hardhat']     += 1
                elif 'mask'    in cn and 'no' not in cn:  detections['mask']        += 1
                elif 'vest'    in cn and 'no' not in cn:  detections['safety_vest'] += 1
                elif 'person'  in cn:                      detections['person']      += 1
                elif 'no-hardhat' in cn or 'no hardhat' in cn:
                    detections['no_hardhat'] += 1
                    detections['violations'].append({'type':'No Hardhat','confidence':conf,'bbox':[x1,y1,x2,y2]})
                elif 'no-mask' in cn or 'no mask' in cn:
                    detections['no_mask'] += 1
                    detections['violations'].append({'type':'No Mask','confidence':conf,'bbox':[x1,y1,x2,y2]})
        return frame, detections

    def calculate_compliance_score(self, detections):
        total = detections.get('person', 0)
        if total == 0:
            return 100.0
        violations = detections['no_hardhat'] + detections['no_mask']
        return max(0.0, (total - violations) / total * 100)

    def log_detection(self, detections, score):
        self.detection_history.append({
            'timestamp':        datetime.now(),
            'hardhat':          detections['hardhat'],
            'mask':             detections['mask'],
            'safety_vest':      detections['safety_vest'],
            'person':           detections['person'],
            'violations':       len(detections['violations']),
            'compliance_score': score,
        })

    def get_analytics_data(self):
        return pd.DataFrame(list(self.detection_history)) if self.detection_history else None


# ── UI helpers ────────────────────────────────────────────────────────────────
def metric_card(icon_svg, value, label, accent="orange"):
    return (f'<div class="metric-card {accent}">'
            f'<div class="metric-icon">{icon_svg}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div></div>')

def alert_banner(level, text):
    icon = ICO_CHECK if level == "success" else ICO_WARN
    return f'<div class="alert-banner alert-{level}">{icon}&nbsp; {text}</div>'

# Build chart layout without duplicate keyword (the root cause of the TypeError)
def make_chart_layout(**overrides):
    base = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Space Mono, monospace', color='#7b8090', size=11),
        xaxis=dict(gridcolor='#1e2230', linecolor='#252836', tickfont=dict(color='#7b8090')),
        yaxis=dict(gridcolor='#1e2230', linecolor='#252836', tickfont=dict(color='#7b8090')),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#b0b6c8')),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    base.update(overrides)   # overrides replaces keys cleanly – no duplicate kwargs
    return base


# ── WebRTC Video Processor ────────────────────────────────────────────────────
class PPEVideoProcessor(VideoProcessorBase):
    """Processes each webcam frame through YOLO PPE detection.

    Runs in a background thread managed by streamlit-webrtc.
    Pushes detection results to a thread-safe queue so the main
    Streamlit thread can display metrics and fire email alerts.
    """

    def __init__(self):
        self.confidence_threshold = 0.5
        self.result_queue: queue.Queue = queue.Queue(maxsize=10)
        self._model = None
        self._ppe_system: PPEDetectionSystem | None = None

    def set_ppe_system(self, ppe_system: PPEDetectionSystem, confidence: float):
        """Called from the main thread after the streamer is created."""
        self._ppe_system = ppe_system
        self._model = ppe_system.model
        self.confidence_threshold = confidence

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        if self._ppe_system and self._model:
            annotated, detections = self._ppe_system.detect_objects(
                img, self.confidence_threshold
            )
            score = self._ppe_system.calculate_compliance_score(detections)

            # Push result; drop oldest if full (non-blocking)
            result = {"detections": detections, "score": score, "frame": annotated}
            try:
                self.result_queue.put_nowait(result)
            except queue.Full:
                try:
                    self.result_queue.get_nowait()  # discard oldest
                except queue.Empty:
                    pass
                self.result_queue.put_nowait(result)

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        return frame


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    st.markdown('<div class="main-header">PPE Safety Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subheader">Real-time workplace safety monitoring</div>',
                unsafe_allow_html=True)

    if 'ppe_system'    not in st.session_state: st.session_state.ppe_system    = PPEDetectionSystem()
    if 'model_loaded'  not in st.session_state: st.session_state.model_loaded  = False

    sys = st.session_state.ppe_system

    # ── Auto-load model on first run (no visible output) ──────────────────────
    if not st.session_state.model_loaded:
        ok = sys.load_model(str(BASE_DIR / "Model" / "ppe.pt"))
        st.session_state.model_loaded = ok

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Control Panel")

        # Model
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">Model</div>', unsafe_allow_html=True)
        s_cls = "status-ok" if sys.model else "status-err"
        s_txt = "Loaded"    if sys.model else "Not Loaded"
        st.markdown(f'<span class="status-pill {s_cls}">{s_txt}</span>', unsafe_allow_html=True)
        if st.button("Reload Model", use_container_width=True):
            with st.spinner("Loading…"):
                ok = sys.load_model(str(BASE_DIR / "Model" / "ppe.pt"))
                st.session_state.model_loaded = ok
                if ok:
                    st.success("Model loaded!")
        st.markdown('</div>', unsafe_allow_html=True)

        # Alert settings
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">Alert Settings</div>', unsafe_allow_html=True)
        email_alerts = st.checkbox("Enable Email Alerts", value=True)
        cooldown_sec = st.slider(
            "Email Interval (seconds)", 10, 600, 60, 10,
            help="Minimum gap between emails for the same violation type.")
        alert_compliance_threshold = st.slider(
            "Alert below compliance (%)", 10, 100, 70, 5,
            help="Emails are sent only when compliance drops below this value.")
        confidence_threshold = st.slider(
            "Min Detection Confidence", 0.1, 1.0, 0.5, 0.05,
            help="Detections below this confidence are ignored.")
        e_cls = "status-ok" if all(sys.email_config.values()) else "status-err"
        e_txt = "Configured" if all(sys.email_config.values()) else "Missing .env"
        st.markdown(f'<span class="status-pill {e_cls}">{e_txt}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Cooldown status
        if sys._email_last_sent:
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-section-title">Last Alerts</div>', unsafe_allow_html=True)
            for vtype, last_sent in sys._email_last_sent.items():
                remaining = max(0, cooldown_sec - int((datetime.now() - last_sent).total_seconds()))
                st.markdown(f"**{vtype}** — {'cooldown: `'+str(remaining)+'s`' if remaining else 'ready'}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Photo storage info
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">Photo Storage</div>', unsafe_allow_html=True)
        photo_files = [f for f in os.listdir(PHOTO_DIR) if f.endswith('.jpg')]
        n_photos    = len(photo_files)
        st.markdown(f"Saved: **{n_photos}** photos in `Photo-Data/`")
        if n_photos > 0:
            if st.button("Clear All Photos"):
                for f in photo_files:
                    try:
                        os.remove(os.path.join(PHOTO_DIR, f))
                    except OSError:
                        pass
                st.success(f"Deleted {n_photos} photo(s).")
                st.rerun()
        else:
            st.caption("No photos saved yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["🎥 Live Detection", "📊 Analytics", "📋 Check Log", "📁 File Upload"])

    # ── Tab 1: Live Detection (WebRTC) ─────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-title">Browser Camera Feed</div>', unsafe_allow_html=True)

        if not sys.model:
            st.warning("Model not loaded. Click **Reload Model** in the sidebar.")
        else:
            st.caption("Click **START** below to open your browser webcam. "
                       "Your browser will ask for camera permission.")

            # Create the WebRTC streamer
            ctx = webrtc_streamer(
                key="ppe-detection",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=PPEVideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            )

            # Inject the PPE system + confidence into the processor once it's alive
            if ctx.video_processor:
                ctx.video_processor.set_ppe_system(sys, confidence_threshold)

            # ── Live metrics & alerts area ────────────────────────────────────
            metrics_ph = st.empty()
            alert_ph   = st.empty()

            if ctx.state.playing and ctx.video_processor:
                # Drain the result queue and show latest detections
                latest = None
                try:
                    while True:
                        latest = ctx.video_processor.result_queue.get_nowait()
                except queue.Empty:
                    pass

                if latest:
                    detections = latest["detections"]
                    score      = latest["score"]
                    sys.log_detection(detections, score)

                    with metrics_ph.container():
                        st.markdown(
                            '<div class="metrics-row">' +
                            metric_card(ICO_PEOPLE,  detections['person'],     "People",       "blue")   +
                            metric_card(ICO_HARDHAT, detections['hardhat'],     "Hardhats",     "orange") +
                            metric_card(ICO_MASK,    detections['mask'],        "Masks",        "green")  +
                            metric_card(ICO_VEST,    detections['safety_vest'], "Safety Vests", "purple") +
                            metric_card(ICO_CHECK,   f"{score:.0f}%",           "Compliance",
                                        "green" if score >= 90 else "red") +
                            '</div>', unsafe_allow_html=True)

                    if   score >= 90: level, msg = "success", "Excellent Safety Compliance!"
                    elif score >= 70: level, msg = "warning", "Good Compliance – minor issues."
                    else:             level, msg = "danger",  "Poor Compliance – immediate action required!"
                    alert_ph.markdown(alert_banner(level, msg), unsafe_allow_html=True)

                    # Email alerts: only when BOTH cooldown elapsed AND compliance < threshold
                    if email_alerts and detections['violations'] and score < alert_compliance_threshold:
                        frame_bgr = latest.get("frame")
                        for v in detections['violations']:
                            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
                            img_path = os.path.join(PHOTO_DIR,
                                                    f"violation_{v['type'].replace(' ','_')}_{ts}.jpg")
                            if frame_bgr is not None:
                                cv2.imwrite(img_path, frame_bgr)
                            threading.Thread(
                                target=sys.send_email_alert,
                                args=(img_path, v['type'], v['confidence'], cooldown_sec),
                                daemon=True).start()

    # ── Tab 2: Analytics ──────────────────────────────────────────────────────
    with tab2:
        df = sys.get_analytics_data()

        if df is not None and not df.empty:
            avg_score  = df['compliance_score'].mean()
            total_viol = int(df['violations'].sum())
            peak_occ   = int(df['person'].max())
            safe_frames = int((df['compliance_score'] >= 90).sum())
            unsafe_frames = int((df['compliance_score'] < 70).sum())
            total_frames = len(df)

            # ── Safety Grade ─────────────────────────────────────────────────
            if avg_score >= 90:
                grade, grade_color, grade_bg, grade_msg = "A", "#4caf83", "#122a1a", "Excellent — site is well protected"
            elif avg_score >= 75:
                grade, grade_color, grade_bg, grade_msg = "B", "#8bc34a", "#1a2a12", "Good — minor issues to address"
            elif avg_score >= 60:
                grade, grade_color, grade_bg, grade_msg = "C", "#ffc107", "#2a2212", "Fair — several workers missing PPE"
            elif avg_score >= 45:
                grade, grade_color, grade_bg, grade_msg = "D", "#ff6b35", "#2a1a12", "Poor — action required immediately"
            else:
                grade, grade_color, grade_bg, grade_msg = "F", "#e05555", "#2a1212", "Danger — serious safety risk on site"

            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:1.5rem;background:{grade_bg};
                        border:2px solid {grade_color};border-radius:14px;
                        padding:1.2rem 1.8rem;margin-bottom:1.5rem;">
                <div style="font-family:'Space Mono',monospace;font-size:4rem;font-weight:700;
                            color:{grade_color};line-height:1;min-width:70px;text-align:center;">
                    {grade}
                </div>
                <div>
                    <div style="font-family:'Space Mono',monospace;font-size:1rem;font-weight:700;
                                color:{grade_color};letter-spacing:0.05em;">
                        SITE SAFETY GRADE
                    </div>
                    <div style="font-size:0.95rem;color:#e8eaf0;margin-top:0.3rem;">{grade_msg}</div>
                    <div style="font-size:0.8rem;color:#7b8090;margin-top:0.2rem;">
                        Based on {total_frames} recorded checks — average compliance: <b style="color:{grade_color}">{avg_score:.0f}%</b>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Plain-English Summary Cards ───────────────────────────────────
            st.markdown('<div class="section-title">Quick Summary</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="metrics-row">' +
                metric_card(ICO_WARN,   total_viol,
                            "Safety Violations Caught",  "red")    +
                metric_card(ICO_CHECK,  safe_frames,
                            "Checks Passed (≥90%)",      "green")  +
                metric_card(ICO_WARN,   unsafe_frames,
                            "Checks Failed (<70%)",      "orange") +
                metric_card(ICO_PEAK,   peak_occ,
                            "Most Workers at Once",      "blue")   +
                '</div>', unsafe_allow_html=True)

            # ── Two-column charts ─────────────────────────────────────────────
            col_l, col_r = st.columns(2)

            with col_l:
                # Simple compliance timeline with colour bands
                st.markdown('<div class="section-title">Was the Site Safe? (Over Time)</div>',
                            unsafe_allow_html=True)
                st.caption("🟢 Green zone = Safe  |  🟡 Yellow = Watch out  |  🔴 Red = Danger")

                fig_t = go.Figure()
                # Coloured background bands
                fig_t.add_hrect(y0=90,  y1=105, fillcolor="rgba(76,175,131,0.07)",  line_width=0)
                fig_t.add_hrect(y0=70,  y1=90,  fillcolor="rgba(255,193,7,0.06)",   line_width=0)
                fig_t.add_hrect(y0=0,   y1=70,  fillcolor="rgba(224,85,85,0.07)",   line_width=0)
                # Compliance line
                fig_t.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['compliance_score'],
                    mode='lines', fill='tozeroy',
                    line=dict(color='#4caf83', width=2),
                    fillcolor='rgba(76,175,131,0.06)',
                    name='Compliance %',
                    hovertemplate='%{x|%H:%M:%S}<br>Compliance: <b>%{y:.0f}%</b><extra></extra>'))
                # Reference lines
                fig_t.add_hline(y=90, line_dash="dot", line_color="#4caf83",
                                annotation_text="Safe (90%)", annotation_font_color="#4caf83",
                                annotation_position="right")
                fig_t.add_hline(y=70, line_dash="dot", line_color="#ffc107",
                                annotation_text="Warning (70%)", annotation_font_color="#ffc107",
                                annotation_position="right")
                fig_t.update_layout(make_chart_layout(
                    yaxis=dict(gridcolor='#1e2230', linecolor='#252836',
                               tickfont=dict(color='#7b8090'), range=[0, 105],
                               ticksuffix='%'),
                    xaxis=dict(gridcolor='#1e2230', linecolor='#252836',
                               tickfont=dict(color='#7b8090')),
                    showlegend=False, height=300))
                st.plotly_chart(fig_t, use_container_width=True)

            with col_r:
                # Simple bar: how many workers had each PPE item on average
                st.markdown('<div class="section-title">Average PPE Worn Per Check</div>',
                            unsafe_allow_html=True)
                st.caption("Shows how many workers were wearing each piece of safety equipment on average")

                ppe_labels  = ['Hard Hat', 'Face Mask', 'Safety Vest', 'Workers on Site']
                ppe_values  = [
                    round(df['hardhat'].mean(), 1),
                    round(df['mask'].mean(), 1),
                    round(df['safety_vest'].mean(), 1),
                    round(df['person'].mean(), 1),
                ]
                ppe_colors  = ['#ff6b35', '#4caf83', '#9b72e8', '#5b8def']

                fig_b = go.Figure(go.Bar(
                    x=ppe_values,
                    y=ppe_labels,
                    orientation='h',
                    marker_color=ppe_colors,
                    text=[f"{v}" for v in ppe_values],
                    textposition='outside',
                    textfont=dict(color='#e8eaf0', size=13),
                    hovertemplate='%{y}: <b>%{x}</b> avg<extra></extra>'))
                fig_b.update_layout(make_chart_layout(
                    xaxis=dict(gridcolor='#1e2230', linecolor='#252836',
                               tickfont=dict(color='#7b8090'), showgrid=True),
                    yaxis=dict(gridcolor='rgba(0,0,0,0)', linecolor='#252836',
                               tickfont=dict(color='#e8eaf0', size=12), autorange='reversed'),
                    showlegend=False, height=300,
                    margin=dict(l=10, r=60, t=20, b=40)))
                st.plotly_chart(fig_b, use_container_width=True)

            # ── Violations breakdown ──────────────────────────────────────────
            st.markdown('<div class="section-title">When Did Violations Happen?</div>',
                        unsafe_allow_html=True)
            st.caption("Red spikes = moments when workers were missing PPE. Aim for a flat line at zero.")

            fig_v = go.Figure()
            fig_v.add_trace(go.Bar(
                x=df['timestamp'], y=df['violations'],
                marker_color=['#e05555' if v > 0 else '#252836' for v in df['violations']],
                name='Violations',
                hovertemplate='%{x|%H:%M:%S}<br>Violations: <b>%{y}</b><extra></extra>'))
            fig_v.update_layout(make_chart_layout(
                yaxis=dict(gridcolor='#1e2230', linecolor='#252836',
                           tickfont=dict(color='#7b8090'), title='Number of Violations',
                           title_font=dict(color='#7b8090', size=11)),
                showlegend=False, height=220))
            st.plotly_chart(fig_v, use_container_width=True)



        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;color:#7b8090;">
                <div style="font-size:3rem;margin-bottom:1rem;">📊</div>
                <div style="font-family:'Space Mono',monospace;font-size:1rem;color:#b0b6c8;margin-bottom:0.5rem;">
                    No data yet
                </div>
                <div style="font-size:0.85rem;">
                    Go to <b style="color:#ff6b35">Live Detection</b> and press <b style="color:#ff6b35">Start</b>
                    — analytics will appear here automatically.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Tab 3: Check Log ──────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-title">Detailed Check Log</div>', unsafe_allow_html=True)
        df_log = sys.get_analytics_data()
        if df_log is not None and not df_log.empty:
            display_df = df_log[['timestamp','person','hardhat','mask','safety_vest',
                                  'violations','compliance_score']].copy()
            display_df.columns = ['Time', 'Workers', 'Hard Hats', 'Masks',
                                   'Safety Vests', 'Violations', 'Safety %']
            display_df['Time'] = display_df['Time'].dt.strftime('%H:%M:%S')
            display_df['Safety %'] = display_df['Safety %'].round(0).astype(int).astype(str) + '%'
            st.dataframe(display_df.sort_values('Time', ascending=False).reset_index(drop=True),
                         use_container_width=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;color:#7b8090;">
                <div style="font-size:3rem;margin-bottom:1rem;">📋</div>
                <div style="font-family:'Space Mono',monospace;font-size:1rem;color:#b0b6c8;margin-bottom:0.5rem;">
                    No records yet
                </div>
                <div style="font-size:0.85rem;">
                    Go to <b style="color:#ff6b35">Live Detection</b> and press
                    <b style="color:#ff6b35">Start</b> — the log will appear here automatically.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Tab 4: File Upload ────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-title">Image &amp; Video Analysis</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload an image or video file",
            type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov', 'mkv'],
            help="Supported: JPG, PNG for images — MP4, AVI, MOV, MKV for videos"
        )

        if not sys.model:
            st.warning("Model not loaded. Check the sidebar.")

        elif uploaded:
            file_ext = uploaded.name.rsplit('.', 1)[-1].lower()
            is_video = file_ext in ('mp4', 'avi', 'mov', 'mkv')

            # ── IMAGE ─────────────────────────────────────────────────────────
            if not is_video:
                image     = Image.open(uploaded)
                img_array = np.array(image)
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                annotated, detections = sys.detect_objects(img_array, confidence_threshold)
                score = sys.calculate_compliance_score(detections)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown('<div class="section-title">Original</div>', unsafe_allow_html=True)
                    st.image(image, use_container_width=True)
                with col_b:
                    st.markdown('<div class="section-title">Detections</div>', unsafe_allow_html=True)
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

                st.markdown(
                    '<div class="metrics-row">' +
                    metric_card(ICO_PEOPLE,  detections['person'],         "People",     "blue")   +
                    metric_card(ICO_HARDHAT, detections['hardhat'],         "Hardhats",   "orange") +
                    metric_card(ICO_VIOL,    len(detections['violations']), "Violations", "red")    +
                    metric_card(ICO_CHECK,   f"{score:.0f}%",               "Compliance",
                                "green" if score >= 90 else "red") +
                    '</div>', unsafe_allow_html=True)

                if detections['violations']:
                    st.markdown(alert_banner("danger",
                        f"{len(detections['violations'])} violation(s) detected."), unsafe_allow_html=True)
                else:
                    st.markdown(alert_banner("success", "No violations detected."), unsafe_allow_html=True)

            # ── VIDEO ─────────────────────────────────────────────────────────
            else:
                import tempfile

                # Save upload to a temp file so OpenCV can read it
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_in:
                    tmp_in.write(uploaded.read())
                    tmp_in_path = tmp_in.name

                cap = cv2.VideoCapture(tmp_in_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps          = cap.get(cv2.CAP_PROP_FPS) or 25
                width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                tmp_out_path = tmp_in_path.replace(f".{file_ext}", "_annotated.mp4")
                fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
                writer  = cv2.VideoWriter(tmp_out_path, fourcc, fps, (width, height))

                st.markdown('<div class="section-title">Processing Video…</div>', unsafe_allow_html=True)
                progress_bar = st.progress(0, text="Starting…")

                all_detections = []
                frame_idx = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    annotated, detections = sys.detect_objects(frame, confidence_threshold)
                    writer.write(annotated)
                    all_detections.append(detections)
                    frame_idx += 1
                    pct = int(frame_idx / max(total_frames, 1) * 100)
                    progress_bar.progress(pct, text=f"Processing frame {frame_idx} of {total_frames}…")

                cap.release()
                writer.release()
                progress_bar.empty()

                # Aggregate stats across all frames
                total_people     = max((d['person']       for d in all_detections), default=0)
                total_hardhats   = max((d['hardhat']      for d in all_detections), default=0)
                total_violations = sum(len(d['violations']) for d in all_detections)
                avg_score        = (sum(
                    sys.calculate_compliance_score(d) for d in all_detections
                ) / max(len(all_detections), 1))

                # Show annotated video
                st.markdown('<div class="section-title">Annotated Video</div>', unsafe_allow_html=True)
                with open(tmp_out_path, 'rb') as vf:
                    st.video(vf.read())

                # Metrics
                st.markdown(
                    '<div class="metrics-row">' +
                    metric_card(ICO_PEOPLE,  total_people,     "Peak People",        "blue")   +
                    metric_card(ICO_HARDHAT, total_hardhats,   "Peak Hardhats",      "orange") +
                    metric_card(ICO_VIOL,    total_violations, "Total Violations",   "red")    +
                    metric_card(ICO_CHECK,   f"{avg_score:.0f}%", "Avg Compliance",
                                "green" if avg_score >= 90 else "red") +
                    metric_card(ICO_FRAMES,  frame_idx,        "Frames Processed",   "purple") +
                    '</div>', unsafe_allow_html=True)

                if total_violations:
                    st.markdown(alert_banner("danger",
                        f"{total_violations} violation(s) detected across {frame_idx} frames."),
                        unsafe_allow_html=True)
                else:
                    st.markdown(alert_banner("success",
                        "No violations detected throughout the video."), unsafe_allow_html=True)

                # Cleanup temp files
                try:
                    os.remove(tmp_in_path)
                    os.remove(tmp_out_path)
                except OSError:
                    pass


if __name__ == "__main__":
    main()
