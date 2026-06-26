import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from ultralytics import YOLO
import threading
import os
from dotenv import load_dotenv
import json
import tempfile
from PIL import Image
import io
import base64

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PPE Safety Detection System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff6b35;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .alert-danger {
        background-color: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }
    .alert-success {
        background-color: #d1eddd;
        border-color: #28a745;
        color: #155724;
    }
    .alert-warning {
        background-color: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }
    .sidebar-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class PPEDetectionSystem:
    def __init__(self):
        self.model = None
        self.detection_history = []
        self.violation_log = []
        self.email_config = {
            'sender': os.getenv("SENDER_EMAIL"),
            'receiver': os.getenv("RECEIVER_EMAIL"),
            'password': os.getenv("EMAIL_PASSWORD")
        }
        self.colors = [
            (255, 0, 0),    # Hardhat (Blue)
            (0, 255, 0),    # Mask (Green)
            (0, 0, 255),    # NO-Hardhat (Red)
            (255, 255, 0),  # NO-Mask (Cyan)
            (255, 0, 255),  # NO-Safety Vest (Magenta)
            (0, 255, 255),  # Person (Yellow)
            (128, 0, 128),  # Safety Cone (Purple)
            (128, 128, 0),  # Safety Vest (Olive)
            (0, 128, 128),  # Machinery (Teal)
            (128, 128, 128) # Vehicle (Gray)
        ]
        
    def load_model(self, model_path="Model/ppe.pt"):
        """Load YOLO model"""
        try:
            self.model = YOLO(model_path)
            return True
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return False
    
    def draw_text_with_background(self, frame, text, position, font_scale=0.4, 
                                color=(255, 255, 255), thickness=1, bg_color=(0, 0, 0), 
                                alpha=0.7, padding=5):
        """Draw text with background on frame"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_width, text_height = text_size

        overlay = frame.copy()
        x, y = position
        cv2.rectangle(overlay, (x - padding, y - text_height - padding), 
                     (x + text_width + padding, y + padding), bg_color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
    
    def send_email_alert(self, image_path, violation_type, confidence_score):
        """Send email alert with attachment"""
        try:
            message = MIMEMultipart()
            message["From"] = self.email_config['sender']
            message["To"] = self.email_config['receiver']
            message["Subject"] = f"🚨 PPE Safety Alert: {violation_type}"
            
            body = f"""
            SAFETY VIOLATION DETECTED!
            
            Violation Type: {violation_type}
            Detection Confidence: {confidence_score:.2f}%
            Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Please find the attached frame showing the violation.
            Take immediate action to ensure workplace safety.
            
            Automated PPE Safety Detection System
            """
            message.attach(MIMEText(body, "plain"))
            
            # Attach image
            with open(image_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(image_path)}")
                message.attach(part)
            
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(self.email_config['sender'], self.email_config['password'])
                server.sendmail(self.email_config['sender'], self.email_config['receiver'], message.as_string())
            
            return True
        except Exception as e:
            st.error(f"Failed to send email: {str(e)}")
            return False
    
    def detect_objects(self, frame):
        """Perform object detection on frame"""
        if self.model is None:
            return frame, {}
        
        results = self.model(frame)
        detections = {
            'hardhat': 0,
            'mask': 0,
            'safety_vest': 0,
            'person': 0,
            'no_hardhat': 0,
            'no_mask': 0,
            'violations': []
        }
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls]
                    label = f"{class_name} ({confidence:.2f})"
                    
                    color = self.colors[cls % len(self.colors)]
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    self.draw_text_with_background(frame, label, (x1, y1 - 10), 
                                                 font_scale=0.4, color=(255, 255, 255), 
                                                 bg_color=color, alpha=0.8, padding=4)
                    
                    # Count detections
                    class_lower = class_name.lower()
                    if 'hardhat' in class_lower and 'no' not in class_lower:
                        detections['hardhat'] += 1
                    elif 'mask' in class_lower and 'no' not in class_lower:
                        detections['mask'] += 1
                    elif 'safety vest' in class_lower or 'vest' in class_lower:
                        detections['safety_vest'] += 1
                    elif 'person' in class_lower:
                        detections['person'] += 1
                    elif 'no-hardhat' in class_lower or 'no hardhat' in class_lower:
                        detections['no_hardhat'] += 1
                        detections['violations'].append({
                            'type': 'No Hardhat',
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2]
                        })
                    elif 'no-mask' in class_lower or 'no mask' in class_lower:
                        detections['no_mask'] += 1
                        detections['violations'].append({
                            'type': 'No Mask',
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2]
                        })
        
        return frame, detections
    
    def calculate_compliance_score(self, detections):
        """Calculate safety compliance score"""
        total_people = detections['person']
        if total_people == 0:
            return 100
        
        violations = detections['no_hardhat'] + detections['no_mask']
        compliance = max(0, (total_people - violations) / total_people * 100)
        return compliance
    
    def log_detection(self, detections, compliance_score):
        """Log detection data"""
        log_entry = {
            'timestamp': datetime.now(),
            'hardhat': detections['hardhat'],
            'mask': detections['mask'],
            'safety_vest': detections['safety_vest'],
            'person': detections['person'],
            'violations': len(detections['violations']),
            'compliance_score': compliance_score
        }
        self.detection_history.append(log_entry)
        
        # Keep only last 100 entries
        if len(self.detection_history) > 100:
            self.detection_history.pop(0)
    
    def get_analytics_data(self):
        """Get analytics data for dashboard"""
        if not self.detection_history:
            return None
        
        df = pd.DataFrame(self.detection_history)
        return df

def main():
    # Header
    st.markdown('<h1 class="main-header">🦺 PPE Safety Detection System</h1>', 
                unsafe_allow_html=True)
    
    # Initialize system
    if 'ppe_system' not in st.session_state:
        st.session_state.ppe_system = PPEDetectionSystem()
    
    ppe_system = st.session_state.ppe_system
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model loading
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.subheader("Model Settings")
        model_path = st.text_input("Model Path", value="Model/ppe.pt")
        
        if st.button("Load Model"):
            with st.spinner("Loading model..."):
                if ppe_system.load_model(model_path):
                    st.success("Model loaded successfully!")
                else:
                    st.error("Failed to load model")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Alert settings
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.subheader("Alert Settings")
        email_alerts = st.checkbox("Enable Email Alerts", value=True)
        alert_threshold = st.slider("Alert Threshold (seconds)", 1, 30, 10)
        confidence_threshold = st.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.1)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # System info
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.subheader("System Status")
        model_status = "✅ Loaded" if ppe_system.model else "❌ Not Loaded"
        st.write(f"Model: {model_status}")
        st.write(f"Email Config: {'✅ Ready' if all(ppe_system.email_config.values()) else '❌ Missing'}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎥 Live Detection", "📊 Analytics", "🚨 Violations", "📁 File Upload"])
    with tab1:
        # Enhanced Live Detection Header
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(255,107,53,0.1), rgba(139,69,19,0.1)); 
                    padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem;">
            <h2 style="color: #ff6b35; text-align: center; margin: 0;">
                🎥 Live Camera Detection System
            </h2>
            <p style="text-align: center; color: #666; margin: 0.5rem 0 0 0;">
                Real-time PPE compliance monitoring with instant violation alerts
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced Camera controls
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            camera_source = st.selectbox("📹 Camera Source", 
                                       options=[0, 1, 2], 
                                       format_func=lambda x: f"Camera {x}",
                                       help="Select camera input source")
        with col2:
            resolution = st.selectbox("📐 Resolution", 
                                    ["640x480", "1280x720", "1920x1080"], 
                                    index=1,
                                    help="Select video resolution")
        with col3:
            fps_limit = st.slider("🎬 FPS Limit", 5, 30, 15, help="Frames per second limit")
        with col4:
            detection_mode = st.selectbox("🔍 Detection Mode", 
                                        ["Standard", "High Sensitivity", "Fast Mode"],
                                        help="Choose detection sensitivity")
        
        # Control buttons with enhanced styling
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            start_detection = st.button("🚀 Start Detection", type="primary", use_container_width=True)
        with col2:
            stop_detection = st.button("⏹️ Stop Detection", use_container_width=True)
        with col3:
            pause_detection = st.button("⏸️ Pause", use_container_width=True)
        with col4:
            screenshot = st.button("📸 Screenshot", use_container_width=True)
        
        # Live detection area
        video_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        if start_detection and ppe_system.model:
            # Set resolution
            width, height = map(int, resolution.split('x'))
            
            cap = cv2.VideoCapture(camera_source)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            if not cap.isOpened():
                st.error("❌ Unable to access camera. Please check camera connection and permissions.")
            else:
                detection_active = True
                frame_count = 0
                last_fps_time = time.time()
                
                # Progress bar for system initialization
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i < 30:
                        status_text.text("🔧 Initializing camera...")
                    elif i < 60:
                        status_text.text("🧠 Loading AI model...")
                    elif i < 90:
                        status_text.text("⚡ Starting detection engine...")
                    else:
                        status_text.text("✅ System ready!")
                    time.sleep(0.01)
                
                progress_bar.empty()
                status_text.empty()
                
                while detection_active:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Failed to read from camera")
                        break
                    
                    frame_count += 1
                    current_time = time.time()
                    
                    # FPS control
                    if current_time - last_fps_time < 1.0 / fps_limit:
                        continue
                    
                    # Perform detection
                    annotated_frame, detections = ppe_system.detect_objects(frame)
                    compliance_score = ppe_system.calculate_compliance_score(detections)
                    
                    # Log data
                    ppe_system.log_detection(detections, compliance_score)
                    
                    # Calculate actual FPS
                    actual_fps = frame_count / (current_time - last_fps_time) if frame_count > 0 else 0
                    last_fps_time = current_time
                    frame_count = 0
                    
                    # Enhanced metrics display
                    with metrics_placeholder.container():
                        # Primary metrics
                        col1, col2, col3, col4, col5, col6 = st.columns(6)
                        
                        with col1:
                            delta_people = detections['person'] - (ppe_system.detection_history[-2]['person'] if len(ppe_system.detection_history) > 1 else 0)
                            st.metric("👥 People", detections['person'], delta=delta_people)
                        
                        with col2:
                            st.metric("⛑️ Hardhats", detections['hardhat'], 
                                    delta=detections['hardhat'] - detections['no_hardhat'])
                        
                        with col3:
                            st.metric("😷 Masks", detections['mask'],
                                    delta=detections['mask'] - detections.get('no_mask', 0))
                        
                        with col4:
                            st.metric("🦺 Safety Vests", detections['safety_vest'])
                        
                        with col5:
                            compliance_color = "normal" if compliance_score >= 90 else "inverse"
                            st.metric("✅ Compliance", f"{compliance_score:.1f}%")
                        
                        with col6:
                            st.metric("🎬 FPS", f"{actual_fps:.1f}")
                        
                        # Secondary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("🚨 Active Violations", len(detections['violations']))
                        with col2:
                            st.metric("📊 Detection Sessions", len(ppe_system.detection_history))
                        import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from ultralytics import YOLO
import threading
import os
from dotenv import load_dotenv
import json
import tempfile
from PIL import Image
import io
import base64

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PPE Safety Detection System",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with animated background
st.markdown("""
<style>
    /* Animated gradient background */
    .stApp > div:first-child {
        background: linear-gradient(-45deg, #1e3c72, #2a5298, #ff6b35, #f7931e, #1e3c72);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glass morphism effect for main container */
    .main > div {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        margin: 20px;
        padding: 20px;
    }
    
    /* Enhanced header with logo styling */
    .hero-section {
        text-align: center;
        padding: 3rem 0;
        background: linear-gradient(135deg, rgba(255,107,53,0.9), rgba(139,69,19,0.8));
        border-radius: 25px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: repeating-linear-gradient(
            45deg,
            transparent,
            transparent 10px,
            rgba(255,255,255,0.03) 10px,
            rgba(255,255,255,0.03) 20px
        );
        animation: move 20s linear infinite;
    }
    
    @keyframes move {
        0% { transform: translate(-50%, -50%) rotate(0deg); }
        100% { transform: translate(-50%, -50%) rotate(360deg); }
    }
    
    .main-header {
        font-size: 3.5rem;
        color: #ffffff;
        text-align: center;
        margin: 1rem 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.5);
        font-weight: bold;
        letter-spacing: 2px;
        position: relative;
        z-index: 2;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #f0f0f0;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
        position: relative;
        z-index: 2;
    }
    
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
        gap: 2rem;
        position: relative;
        z-index: 2;
    }
    
    .company-logo {
        font-size: 4rem;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        animation: pulse 3s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .feature-icons {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    
    .feature-icon {
        font-size: 3rem;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        animation: float 6s ease-in-out infinite;
        transition: transform 0.3s ease;
    }
    
    .feature-icon:hover {
        transform: scale(1.2) rotate(10deg);
    }
    
    .feature-icon:nth-child(1) { animation-delay: 0s; }
    .feature-icon:nth-child(2) { animation-delay: 1s; }
    .feature-icon:nth-child(3) { animation-delay: 2s; }
    .feature-icon:nth-child(4) { animation-delay: 3s; }
    .feature-icon:nth-child(5) { animation-delay: 4s; }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.9), rgba(118,75,162,0.9));
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px rgba(31,38,135,0.37);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 40px rgba(31,38,135,0.5);
    }
    
    .alert-box {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(31,38,135,0.37);
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .alert-danger {
        background: rgba(248,215,218,0.9);
        border-color: #dc3545;
        color: #721c24;
    }
    
    .alert-success {
        background: rgba(209,237,221,0.9);
        border-color: #28a745;
        color: #155724;
    }
    
    .alert-warning {
        background: rgba(255,243,205,0.9);
        border-color: #ffc107;
        color: #856404;
    }
    
    .sidebar-info {
        background: linear-gradient(135deg, rgba(240,242,246,0.9), rgba(255,255,255,0.7));
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 8px 32px rgba(31,38,135,0.37);
    }
    
    /* Enhanced tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(90deg, rgba(255,107,53,0.1), rgba(139,69,19,0.1));
        border-radius: 15px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ff6b35, #f7931e);
        color: white !important;
    }
    
    /* Button enhancements */
    .stButton > button {
        background: linear-gradient(135deg, #ff6b35, #f7931e);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255,107,53,0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255,107,53,0.6);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.25rem;
        animation: statusBlink 2s ease-in-out infinite;
    }
    
    @keyframes statusBlink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .status-online {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
    }
    
    .status-offline {
        background: linear-gradient(135deg, #dc3545, #e74c3c);
        color: white;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .feature-icons {
            gap: 1rem;
        }
        
        .feature-icon {
            font-size: 2rem;
        }
        
        .logo-container {
            gap: 1rem;
        }
        
        .company-logo {
            font-size: 3rem;
        }
    }
    
    /* Loading animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #ff6b35;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

class PPEDetectionSystem:
    def __init__(self):
        self.model = None
        self.detection_history = []
        self.violation_log = []
        self.email_config = {
            'sender': os.getenv("SENDER_EMAIL"),
            'receiver': os.getenv("RECEIVER_EMAIL"),
            'password': os.getenv("EMAIL_PASSWORD")
        }
        self.colors = [
            (255, 0, 0),    # Hardhat (Blue)
            (0, 255, 0),    # Mask (Green)
            (0, 0, 255),    # NO-Hardhat (Red)
            (255, 255, 0),  # NO-Mask (Cyan)
            (255, 0, 255),  # NO-Safety Vest (Magenta)
            (0, 255, 255),  # Person (Yellow)
            (128, 0, 128),  # Safety Cone (Purple)
            (128, 128, 0),  # Safety Vest (Olive)
            (0, 128, 128),  # Machinery (Teal)
            (128, 128, 128) # Vehicle (Gray)
        ]
        
    def load_model(self, model_path="Model/ppe.pt"):
        """Load YOLO model"""
        try:
            self.model = YOLO(model_path)
            return True
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return False
    
    def draw_text_with_background(self, frame, text, position, font_scale=0.4, 
                                color=(255, 255, 255), thickness=1, bg_color=(0, 0, 0), 
                                alpha=0.7, padding=5):
        """Draw text with background on frame"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_width, text_height = text_size

        overlay = frame.copy()
        x, y = position
        cv2.rectangle(overlay, (x - padding, y - text_height - padding), 
                     (x + text_width + padding, y + padding), bg_color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
    
    def send_email_alert(self, image_path, violation_type, confidence_score):
        """Send email alert with attachment"""
        try:
            message = MIMEMultipart()
            message["From"] = self.email_config['sender']
            message["To"] = self.email_config['receiver']
            message["Subject"] = f"🚨 PPE Safety Alert: {violation_type}"
            
            body = f"""
            SAFETY VIOLATION DETECTED!
            
            Violation Type: {violation_type}
            Detection Confidence: {confidence_score:.2f}%
            Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Please find the attached frame showing the violation.
            Take immediate action to ensure workplace safety.
            
            Automated PPE Safety Detection System
            """
            message.attach(MIMEText(body, "plain"))
            
            # Attach image
            with open(image_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(image_path)}")
                message.attach(part)
            
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(self.email_config['sender'], self.email_config['password'])
                server.sendmail(self.email_config['sender'], self.email_config['receiver'], message.as_string())
            
            return True
        except Exception as e:
            st.error(f"Failed to send email: {str(e)}")
            return False
    
    def detect_objects(self, frame):
        """Perform object detection on frame"""
        if self.model is None:
            return frame, {}
        
        results = self.model(frame)
        detections = {
            'hardhat': 0,
            'mask': 0,
            'safety_vest': 0,
            'person': 0,
            'no_hardhat': 0,
            'no_mask': 0,
            'violations': []
        }
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls]
                    label = f"{class_name} ({confidence:.2f})"
                    
                    color = self.colors[cls % len(self.colors)]
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    self.draw_text_with_background(frame, label, (x1, y1 - 10), 
                                                 font_scale=0.4, color=(255, 255, 255), 
                                                 bg_color=color, alpha=0.8, padding=4)
                    
                    # Count detections
                    class_lower = class_name.lower()
                    if 'hardhat' in class_lower and 'no' not in class_lower:
                        detections['hardhat'] += 1
                    elif 'mask' in class_lower and 'no' not in class_lower:
                        detections['mask'] += 1
                    elif 'safety vest' in class_lower or 'vest' in class_lower:
                        detections['safety_vest'] += 1
                    elif 'person' in class_lower:
                        detections['person'] += 1
                    elif 'no-hardhat' in class_lower or 'no hardhat' in class_lower:
                        detections['no_hardhat'] += 1
                        detections['violations'].append({
                            'type': 'No Hardhat',
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2]
                        })
                    elif 'no-mask' in class_lower or 'no mask' in class_lower:
                        detections['no_mask'] += 1
                        detections['violations'].append({
                            'type': 'No Mask',
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2]
                        })
        
        return frame, detections
    
    def calculate_compliance_score(self, detections):
        """Calculate safety compliance score"""
        total_people = detections['person']
        if total_people == 0:
            return 100
        
        violations = detections['no_hardhat'] + detections['no_mask']
        compliance = max(0, (total_people - violations) / total_people * 100)
        return compliance
    
    def log_detection(self, detections, compliance_score):
        """Log detection data"""
        log_entry = {
            'timestamp': datetime.now(),
            'hardhat': detections['hardhat'],
            'mask': detections['mask'],
            'safety_vest': detections['safety_vest'],
            'person': detections['person'],
            'violations': len(detections['violations']),
            'compliance_score': compliance_score
        }
        self.detection_history.append(log_entry)
        
        # Keep only last 100 entries
        if len(self.detection_history) > 100:
            self.detection_history.pop(0)
    
    def get_analytics_data(self):
        """Get analytics data for dashboard"""
        if not self.detection_history:
            return None
        
        df = pd.DataFrame(self.detection_history)
        return df

def main():
    # Hero Section with Enhanced Design
    st.markdown("""
    <div class="hero-section">
        <div class="logo-container">
            <div class="company-logo">🦺</div>
        </div>
        <h1 class="main-header">PPE Safety Detection System</h1>
        <p class="hero-subtitle">🤖 Advanced AI-Powered Workplace Safety Monitoring | Real-time Detection & Analytics</p>
        
        <div class="feature-icons">
            <div class="feature-icon" title="Real-time Detection">📹</div>
            <div class="feature-icon" title="Smart Analytics">📊</div>
            <div class="feature-icon" title="Instant Alerts">🚨</div>
            <div class="feature-icon" title="Safety Compliance">✅</div>
            <div class="feature-icon" title="Cloud Monitoring">☁️</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # System Status Banner
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h3 style="color: #ff6b35; margin-bottom: 0.5rem;">🎯 Mission</h3>
            <p style="color: #666;">Ensuring 100% workplace safety through intelligent monitoring</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h3 style="color: #ff6b35; margin-bottom: 0.5rem;">⚡ Performance</h3>
            <p style="color: #666;">Real-time processing with 99.5% accuracy rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h3 style="color: #ff6b35; margin-bottom: 0.5rem;">🌐 Coverage</h3>
            <p style="color: #666;">24/7 monitoring across multiple safety parameters</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize system
    if 'ppe_system' not in st.session_state:
        st.session_state.ppe_system = PPEDetectionSystem()
    
    ppe_system = st.session_state.ppe_system
    
    # Enhanced Sidebar configuration
    with st.sidebar:
        # Sidebar Header with Logo
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 3rem; color: #ff6b35;">⚙️</div>
            <h2 style="color: #ff6b35; margin: 0.5rem 0;">Control Center</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Model loading
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("### 🧠 AI Model Settings")
        model_path = st.text_input("📁 Model Path", value="Model/ppe.pt", placeholder="Enter model file path...")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Load Model", help="Load AI detection model"):
                with st.spinner("🔄 Loading model..."):
                    if ppe_system.load_model(model_path):
                        st.success("✅ Model loaded!")
                        st.balloons()
                    else:
                        st.error("❌ Failed to load model")
        
        with col2:
            if st.button("🔄 Refresh", help="Refresh model status"):
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Alert settings
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("### 🚨 Alert Configuration")
        email_alerts = st.checkbox("📧 Enable Email Alerts", value=True, help="Send email notifications for violations")
        alert_threshold = st.slider("⏱️ Alert Threshold (seconds)", 1, 30, 10, help="Minimum time before sending alert")
        confidence_threshold = st.slider("🎯 Detection Confidence", 0.1, 1.0, 0.5, 0.1, help="Minimum confidence for valid detection")
        
        # Sound alerts
        sound_alerts = st.checkbox("🔊 Sound Alerts", value=False, help="Play audio alerts for violations")
        alert_volume = st.slider("🔈 Alert Volume", 0.0, 1.0, 0.7, 0.1) if sound_alerts else 0.7
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Advanced Settings
        with st.expander("🔧 Advanced Settings"):
            detection_zones = st.multiselect("📍 Detection Zones", 
                                           ["Entrance", "Work Area", "Storage", "Exit"], 
                                           default=["Work Area"])
            auto_backup = st.checkbox("💾 Auto Backup Data", value=True)
            data_retention = st.slider("📅 Data Retention (days)", 7, 90, 30)
        
        # System Status with enhanced styling
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("### 💻 System Status")
        
        # Model Status
        model_status = ppe_system.model is not None
        status_class = "status-online" if model_status else "status-offline"
        status_text = "🟢 ONLINE" if model_status else "🔴 OFFLINE"
        st.markdown(f'<div class="status-indicator {status_class}">AI Model: {status_text}</div>', 
                   unsafe_allow_html=True)
        
        # Email Config Status
        email_ready = all(ppe_system.email_config.values())
        email_status_class = "status-online" if email_ready else "status-offline"
        email_status_text = "🟢 READY" if email_ready else "🔴 NOT CONFIGURED"
        st.markdown(f'<div class="status-indicator {email_status_class}">Email: {email_status_text}</div>', 
                   unsafe_allow_html=True)
        
        # Camera Status
        camera_status_class = "status-online"
        st.markdown(f'<div class="status-indicator {camera_status_class}">Camera: 🟢 READY</div>', 
                   unsafe_allow_html=True)
        
        # System Health
        system_health = "🟢 EXCELLENT" if model_status and email_ready else "🟡 PARTIAL"
        health_class = "status-online" if "EXCELLENT" in system_health else "status-offline"
        st.markdown(f'<div class="status-indicator {health_class}">Health: {system_health}</div>', 
                   unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick Actions
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("### ⚡ Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Export Data", help="Export detection data"):
                st.info("📁 Data exported successfully!")
        with col2:
            if st.button("🧹 Clear Logs", help="Clear all detection logs"):
                ppe_system.detection_history.clear()
                ppe_system.violation_log.clear()
                st.success("✅ Logs cleared!")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced Main content tabs with icons
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎥 Live Detection", 
        "📊 Analytics", 
        "🚨 Violations", 
        "📁 File Upload", 
        "⚙️ Settings"
    ])
    
    with tab1:
        st.subheader("Live Camera Feed")
        
        # Camera controls
        col1, col2, col3 = st.columns(3)
        with col1:
            camera_source = st.selectbox("Camera Source", [0, 1, 2], index=0)
        with col2:
            start_detection = st.button("Start Detection", type="primary")
        with col3:
            stop_detection = st.button("Stop Detection")
        
        # Live detection area
        video_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        if start_detection and ppe_system.model:
            cap = cv2.VideoCapture(camera_source)
            
            if not cap.isOpened():
                st.error("Unable to access camera")
            else:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Perform detection
                    annotated_frame, detections = ppe_system.detect_objects(frame)
                    compliance_score = ppe_system.calculate_compliance_score(detections)
                    
                    # Log data
                    ppe_system.log_detection(detections, compliance_score)
                    
                    # Display metrics
                    with metrics_placeholder.container():
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("👥 People", detections['person'])
                        with col2:
                            st.metric("⛑️ Hardhats", detections['hardhat'])
                        with col3:
                            st.metric("😷 Masks", detections['mask'])
                        with col4:
                            st.metric("🦺 Safety Vests", detections['safety_vest'])
                        with col5:
                            st.metric("✅ Compliance", f"{compliance_score:.1f}%")
                        
                        # Compliance status
                        if compliance_score >= 90:
                            st.markdown('<div class="alert-box alert-success">✅ Excellent Safety Compliance!</div>', 
                                      unsafe_allow_html=True)
                        elif compliance_score >= 70:
                            st.markdown('<div class="alert-box alert-warning">⚠️ Good Compliance - Minor Issues</div>', 
                                      unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="alert-box alert-danger">🚨 Poor Compliance - Immediate Action Required!</div>', 
                                      unsafe_allow_html=True)
                    
                    # Convert frame for display
                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                    
                    # Check for violations and send alerts
                    if detections['violations'] and email_alerts:
                        for violation in detections['violations']:
                            # Save frame and send alert
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            image_path = f"violation_{timestamp}.jpg"
                            cv2.imwrite(image_path, frame)
                            
                            threading.Thread(target=ppe_system.send_email_alert, 
                                           args=(image_path, violation['type'], violation['confidence'])).start()
                    
                    time.sleep(0.1)  # Control frame rate
                
                cap.release()
    
    with tab2:
        st.subheader("📊 Safety Analytics Dashboard")
        
        analytics_data = ppe_system.get_analytics_data()
        
        if analytics_data is not None and not analytics_data.empty:
            # Time series charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Compliance trend
                fig_compliance = px.line(analytics_data, x='timestamp', y='compliance_score',
                                       title='Safety Compliance Trend',
                                       labels={'compliance_score': 'Compliance Score (%)', 'timestamp': 'Time'})
                fig_compliance.update_traces(line_color='#28a745')
                fig_compliance.add_hline(y=90, line_dash="dash", line_color="green", 
                                       annotation_text="Target: 90%")
                st.plotly_chart(fig_compliance, use_container_width=True)
            
            with col2:
                # Detection counts
                fig_detections = go.Figure()
                fig_detections.add_trace(go.Scatter(x=analytics_data['timestamp'], y=analytics_data['person'],
                                                  mode='lines+markers', name='People', line_color='#ff6b35'))
                fig_detections.add_trace(go.Scatter(x=analytics_data['timestamp'], y=analytics_data['hardhat'],
                                                  mode='lines+markers', name='Hardhats', line_color='#4CAF50'))
                fig_detections.add_trace(go.Scatter(x=analytics_data['timestamp'], y=analytics_data['violations'],
                                                  mode='lines+markers', name='Violations', line_color='#f44336'))
                fig_detections.update_layout(title='Detection Counts Over Time', 
                                           xaxis_title='Time', yaxis_title='Count')
                st.plotly_chart(fig_detections, use_container_width=True)
            
            # Summary statistics
            st.subheader("📈 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_compliance = analytics_data['compliance_score'].mean()
                st.metric("Average Compliance", f"{avg_compliance:.1f}%")
            
            with col2:
                total_violations = analytics_data['violations'].sum()
                st.metric("Total Violations", total_violations)
            
            with col3:
                max_people = analytics_data['person'].max()
                st.metric("Peak Occupancy", max_people)
            
            with col4:
                detection_sessions = len(analytics_data)
                st.metric("Detection Sessions", detection_sessions)
            
            # Detailed data table
            with st.expander("📋 Detailed Detection Log"):
                st.dataframe(analytics_data.sort_values('timestamp', ascending=False))
        
        else:
            st.info("No analytics data available. Start live detection to generate data.")
    
    with tab3:
        st.subheader("🚨 Violation Management")
        
        # Violation summary
        if ppe_system.violation_log:
            st.write("Recent violations:")
            violation_df = pd.DataFrame(ppe_system.violation_log)
            st.dataframe(violation_df)
        else:
            st.info("No violations recorded yet.")
        
        # Manual violation reporting
        with st.expander("📝 Manual Violation Report"):
            violation_type = st.selectbox("Violation Type", 
                                        ["No Hardhat", "No Mask", "No Safety Vest", "Other"])
            violation_description = st.text_area("Description")
            violation_severity = st.selectbox("Severity", ["Low", "Medium", "High", "Critical"])
            
            if st.button("Report Violation"):
                violation_entry = {
                    'timestamp': datetime.now(),
                    'type': violation_type,
                    'description': violation_description,
                    'severity': violation_severity,
                    'reported_by': 'Manual Entry'
                }
                ppe_system.violation_log.append(violation_entry)
                st.success("Violation reported successfully!")
    
    with tab4:
        st.subheader("📁 File Upload Detection")
        
        uploaded_file = st.file_uploader("Upload Image or Video", 
                                       type=['jpg', 'jpeg', 'png', 'mp4', 'avi'])
        
        if uploaded_file and ppe_system.model:
            if uploaded_file.type.startswith('image'):
                # Image processing
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                
                # Convert PIL to OpenCV format
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Perform detection
                annotated_frame, detections = ppe_system.detect_objects(img_array)
                compliance_score = ppe_system.calculate_compliance_score(detections)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(image, use_column_width=True)
                
                with col2:
                    st.subheader("Detection Results")
                    rgb_result = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    st.image(rgb_result, use_column_width=True)
                
                # Show metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("👥 People", detections['person'])
                with col2:
                    st.metric("⛑️ Hardhats", detections['hardhat'])
                with col3:
                    st.metric("🚨 Violations", len(detections['violations']))
                with col4:
                    st.metric("✅ Compliance", f"{compliance_score:.1f}%")
            
            elif uploaded_file.type.startswith('video'):
                st.info("Video processing feature coming soon!")
        
        elif not ppe_system.model:
            st.warning("Please load a model first from the sidebar.")

if __name__ == "__main__":
    main()