import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import smtplib, ssl, os, datetime, time, json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import streamlit.components.v1 as components

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(page_title="Smart Traffic AI Dashboard", layout="wide")
st.title("🚦 Smart Traffic Management Dashboard")
st.markdown("Emergency Vehicle Detection  |  Accident Detection")

# ----------------------------------
# Email Config (ENV VARIABLES)
# ----------------------------------
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")

EMAIL_COOLDOWN = 300  # 5 minutes

# ----------------------------------
# Session State
# ----------------------------------
if "last_email_time" not in st.session_state:
    st.session_state.last_email_time = 0

if "email_sent" not in st.session_state:
    st.session_state.email_sent = False

if "email_sent_time" not in st.session_state:
    st.session_state.email_sent_time = ""

# ----------------------------------
# Load Models
# ----------------------------------
@st.cache_resource
def load_models():
    emergency_model = YOLO("emergency.pt")
    accident_model = YOLO("best.pt")
    return emergency_model, accident_model

emergency_model, accident_model = load_models()

# ----------------------------------
# Browser Location
# ----------------------------------
def get_browser_location():
    js = """
    <script>
    navigator.geolocation.getCurrentPosition(
        (pos) => {
            const data = {
                lat: pos.coords.latitude,
                lon: pos.coords.longitude,
                time: new Date().toLocaleString()
            };
            document.write(JSON.stringify(data));
        },
        (err) => {
            document.write(JSON.stringify({error: err.message}));
        }
    );
    </script>
    """
    result = components.html(js, height=0)
    try:
        return json.loads(result)
    except:
        return {}

# ----------------------------------
# Email Function
# ----------------------------------
def send_accident_alert(frame, location):
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"accident_{timestamp}.jpg"
        cv2.imwrite(filename, frame)

        lat = location.get("lat", "N/A")
        lon = location.get("lon", "N/A")
        time_str = location.get("time", "N/A")

        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg["Subject"] = "🚨 ACCIDENT ALERT (REAL-TIME)"

        body = f"""
🚨 ACCIDENT DETECTED 🚨

📍 Live Location:
Latitude : {lat}
Longitude: {lon}

🕒 Time:
{time_str}

Immediate emergency response required.
        """
        msg.attach(MIMEText(body, "plain"))

        with open(filename, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={filename}")
        msg.attach(part)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())

        os.remove(filename)
        return True

    except Exception as e:
        st.error(f"❌ Email failed: {e}")
        return False

# ----------------------------------
# Sidebar
# ----------------------------------
st.sidebar.header("⚙️ Controls")
conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)

uploaded_video = st.sidebar.file_uploader(
    "Upload Traffic Video",
    type=["mp4", "avi", "mov", "webm"]
)

# ----------------------------------
# Layout
# ----------------------------------
col1, col2 = st.columns(2)
col1.subheader("🚑 Emergency Vehicle Detection")
col2.subheader("🚨 Accident Detection")

frame_emergency = col1.empty()
frame_accident = col2.empty()
alert_box = col2.empty()
mail_status_box = col2.empty()

# ----------------------------------
# Video Processing
# ----------------------------------
if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Emergency Detection
        res1 = emergency_model(frame, conf=conf)
        out1 = cv2.cvtColor(res1[0].plot(), cv2.COLOR_BGR2RGB)

        # Accident Detection
        res2 = accident_model(frame, conf=conf)
        out2 = cv2.cvtColor(res2[0].plot(), cv2.COLOR_BGR2RGB)

        # Accident Check
        accident_detected = False
        if res2[0].boxes is not None:
            for box in res2[0].boxes:
                if box.conf.item() > 0.6:
                    accident_detected = True
                    break

        current_time = time.time()

        if accident_detected:
            alert_box.error("🚨 ACCIDENT DETECTED")

            if current_time - st.session_state.last_email_time > EMAIL_COOLDOWN:
                location = get_browser_location()
                success = send_accident_alert(frame, location)

                if success:
                    st.session_state.last_email_time = current_time
                    st.session_state.email_sent = True
                    st.session_state.email_sent_time = datetime.datetime.now().strftime("%H:%M:%S")

        else:
            alert_box.success("✅ No Accident Detected")
            st.session_state.email_sent = False

        if st.session_state.email_sent:
            mail_status_box.success(
                f"📧 Alert email sent successfully at {st.session_state.email_sent_time}"
            )

        frame_emergency.image(out1, channels="RGB", use_container_width=True)
        frame_accident.image(out2, channels="RGB", use_container_width=True)

    cap.release()
