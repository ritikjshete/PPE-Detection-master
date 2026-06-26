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
from pathlib import Path
from dotenv import load_dotenv
import json
import tempfile
from PIL import Image
import io
import base64

# Resolve project root and load .env reliably
BASE_DIR = Path(__file__).parent.resolve()
load_dotenv(dotenv_path=BASE_DIR / ".env")

PHOTO_DIR = "Photo-Data"
os.makedirs(PHOTO_DIR, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="PPE Safety Detection System",
    page_icon="🦺",
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
        email_pw = os.getenv("EMAIL_PASSWORD")
        self.email_config = {
            'sender': os.getenv("SENDER_EMAIL"),
            'receiver': os.getenv("RECEIVER_EMAIL"),
            'password': email_pw.strip().replace(" ", "") if email_pw else None
        }
        self._email_last_sent: dict[str, datetime] = {}
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
    
    # ── Throttled email ───────────────────────────────────────────────────────
    def _can_send_email(self, violation_type: str, cooldown_seconds: int) -> bool:
        """Check if enough time has elapsed since the last email for this violation type."""
        last = self._email_last_sent.get(violation_type)
        return last is None or (datetime.now() - last).total_seconds() >= cooldown_seconds

    def send_email_alert(self, image_path, violation_type, confidence_score,
                         cooldown_seconds: int = 60):
        """Send email alert with attachment (respects per-type cooldown)."""
        if not self._can_send_email(violation_type, cooldown_seconds):
            return False
        try:
            message = MIMEMultipart()
            message["From"] = self.email_config['sender']
            message["To"] = self.email_config['receiver']
            message["Subject"] = f"PPE Safety Alert: {violation_type}"
            
            body = f"""
            SAFETY VIOLATION DETECTED!
            
            Violation Type: {violation_type}
            Detection Confidence: {confidence_score * 100:.2f}%
            Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Please find the attached frame showing the violation.
            Take immediate action to ensure workplace safety.
            
            Automated PPE Safety Detection System
            """
            message.attach(MIMEText(body, "plain"))
            
            # Attach image if it exists
            if os.path.exists(image_path):
                with open(image_path, "rb") as attachment:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header("Content-Disposition",
                                    f"attachment; filename={os.path.basename(image_path)}")
                    message.attach(part)
            
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(self.email_config['sender'], self.email_config['password'])
                server.sendmail(self.email_config['sender'],
                                self.email_config['receiver'], message.as_string())
            
            self._email_last_sent[violation_type] = datetime.now()
            return True
        except Exception as e:
            print(f"[Email Error] {e}")
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
                            # Save frame to Photo-Data/ directory
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            image_path = os.path.join(
                                PHOTO_DIR,
                                f"violation_{violation['type'].replace(' ', '_')}_{timestamp}.jpg")
                            cv2.imwrite(image_path, frame)
                            
                            threading.Thread(
                                target=ppe_system.send_email_alert,
                                args=(image_path, violation['type'],
                                      violation['confidence'], alert_threshold),
                                daemon=True).start()
                    
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