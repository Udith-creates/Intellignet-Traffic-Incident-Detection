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
from datetime import datetime as dt
import requests
import asyncio
import sys

# Windows Geolocation imports
try:
    from winsdk.windows.devices.geolocation import Geolocator
    WINSDK_AVAILABLE = True
except ImportError:
    WINSDK_AVAILABLE = False

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

if "current_location" not in st.session_state:
    st.session_state.current_location = None

if "location_fetched" not in st.session_state:
    st.session_state.location_fetched = False

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
# Location Detection Functions
# ----------------------------------
async def get_location_from_windows_gps():
    """Get accurate location using Windows Geolocation API (GPS/WiFi)"""
    if not WINSDK_AVAILABLE:
        return None
    
    try:
        locator = Geolocator()
        locator.desired_accuracy_in_meters = 10
        
        pos = await locator.get_geoposition_async()
        
        lat = pos.coordinate.point.position.latitude
        lon = pos.coordinate.point.position.longitude
        accuracy = pos.coordinate.accuracy
        
        return {
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "city": "Current Location",
            "region": "Via GPS",
            "country": "Windows Device",
            "isp": "Local Network",
            "method": "Windows GPS/WiFi Geolocation",
            "accuracy": f"±{accuracy:.0f} meters"
        }
    except Exception as e:
        print(f"Windows geolocation error: {e}")
        return None

@st.cache_data(ttl=3600)
def get_location_from_ip():
    """Get approximate location using IP address geolocation API"""
    try:
        # Try multiple services for reliability
        services = [
            ("https://ipapi.co/json/", lambda r: {
                "lat": r.get("latitude"),
                "lon": r.get("longitude"),
                "city": r.get("city", "Unknown"),
                "region": r.get("region", ""),
                "country": r.get("country_name", "Unknown"),
                "isp": r.get("org", ""),
            }),
            ("https://ip-api.com/json/", lambda r: {
                "lat": r.get("lat"),
                "lon": r.get("lon"),
                "city": r.get("city", "Unknown"),
                "region": r.get("region", ""),
                "country": r.get("country", "Unknown"),
                "isp": r.get("isp", ""),
            }),
        ]
        
        for url, parser in services:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data and "lat" in data and "lon" in data:
                        location = parser(data)
                        location["method"] = "IP-based Geolocation"
                        location["accuracy"] = "Approximate (City Level)"
                        return location
            except:
                continue
        
        return None
    except Exception as e:
        print(f"IP geolocation error: {e}")
        return None

def get_current_location():
    """Get location with Windows GPS as primary, IP-based as fallback"""
    location = None
    
    # Try Windows GPS first if available
    if WINSDK_AVAILABLE:
        try:
            # Get or create event loop
            if sys.platform == 'win32':
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                location = loop.run_until_complete(get_location_from_windows_gps())
                loop.close()
            else:
                location = asyncio.run(get_location_from_windows_gps())
            
            if location:
                return location
        except Exception as e:
            print(f"Windows GPS async error: {e}")
    
    # Fallback to IP-based location
    location = get_location_from_ip()
    if location:
        return location
    
    # Final fallback when all methods fail
    return {
        "lat": "N/A",
        "lon": "N/A",
        "city": "Unknown",
        "region": "Unknown",
        "country": "Unknown",
        "isp": "Unknown",
        "method": "Location unavailable",
        "accuracy": "None"
    }

def get_current_time():
    """Get current time in proper format"""
    return dt.now().strftime("%Y-%m-%d %H:%M:%S")

# ----------------------------------
# Initialize Current Location at Start
# ----------------------------------
if not st.session_state.location_fetched:
    with st.spinner("📍 Fetching your current location..."):
        st.session_state.current_location = get_current_location()
        st.session_state.location_fetched = True

# Display Location Info
if st.session_state.current_location:
    location = st.session_state.current_location
    col_loc = st.columns(3)
    
    with col_loc[0]:
        st.metric("🗺️ Latitude", f"{location.get('lat', 'N/A')}")
    with col_loc[1]:
        st.metric("🗺️ Longitude", f"{location.get('lon', 'N/A')}")
    with col_loc[2]:
        st.metric("🔍 Method", location.get("method", "Unknown").split()[0])
    
    st.divider()

# ----------------------------------
# Email Function
# ----------------------------------
def send_accident_alert(frame, location, alert_time):
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"accident_{timestamp}.jpg"
        cv2.imwrite(filename, frame)

        lat = location.get("lat", "N/A")
        lon = location.get("lon", "N/A")
        city = location.get("city", "Unknown")
        region = location.get("region", "")
        country = location.get("country", "Unknown")
        isp = location.get("isp", "")
        method = location.get("method", "Unknown")
        accuracy = location.get("accuracy", "Unknown")

        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg["Subject"] = "🚨 ACCIDENT ALERT (REAL-TIME)"

        body = f"""
🚨 ACCIDENT DETECTED 🚨

📍 LOCATION DETAILS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Latitude  : {lat}
Longitude : {lon}

Region    : {region if region else 'N/A'}

ISP       : {isp if isp else 'N/A'}

🔍 DETECTION METHOD:
Method    : {method}
Accuracy  : {accuracy}

🕒 ALERT TIMESTAMP:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{alert_time}

⚠️ URGENT - Immediate emergency response required!
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
            alert_time = get_current_time()

            if current_time - st.session_state.last_email_time > EMAIL_COOLDOWN:
                # Use cached location
                location = st.session_state.current_location if st.session_state.current_location else get_current_location()
                success = send_accident_alert(frame, location, alert_time)

                if success:
                    st.session_state.last_email_time = current_time
                    st.session_state.email_sent = True
                    st.session_state.email_sent_time = dt.now().strftime("%H:%M:%S")

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
