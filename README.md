# 🚦 Smart Traffic Management Dashboard

An AI-powered real-time traffic monitoring system that detects emergency vehicles and traffic accidents using YOLO object detection models with automated email alerts.

## ✨ Features

- **🚑 Emergency Vehicle Detection** - Real-time identification of ambulances, fire trucks, and police vehicles
- **🚨 Accident Detection** - Automatic detection of traffic accidents with visual alerts
- **📧 Email Alerts** - Automated email notifications with accident images and GPS location coordinates
- **🎥 Real-time Video Processing** - Live video analysis with side-by-side detection results
- **🌍 GPS Integration** - Browser-based geolocation capture for accident incidents
- **⚙️ Configurable Confidence** - Adjustable detection confidence threshold (0.1 - 1.0)
- **📱 Web Dashboard** - Interactive Streamlit web interface

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for real-time processing)
- Webcam or video files for input

## 🚀 Installation

1. **Clone or extract the project**
   ```bash
   cd "Intelligent Traffic Incident Detection"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify CUDA installation** (optional but recommended)
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

## ⚙️ Configuration

### Environment Variables

Set the following environment variables for email alerts functionality:

```bash
# Windows PowerShell
$env:SENDER_EMAIL = "your_email@gmail.com"
$env:SENDER_PASSWORD = "your_app_password"
$env:RECEIVER_EMAIL = "alert_recipient@gmail.com"
```

```bash
# Windows CMD
set SENDER_EMAIL=your_email@gmail.com
set SENDER_PASSWORD=your_app_password
set RECEIVER_EMAIL=alert_recipient@gmail.com
```

```bash
# Linux/Mac
export SENDER_EMAIL="your_email@gmail.com"
export SENDER_PASSWORD="your_app_password"
export RECEIVER_EMAIL="alert_recipient@gmail.com"
```

**Note:** For Gmail, use an [App Password](https://myaccount.google.com/apppasswords) instead of your regular password.

## 📖 Usage

### Run Web Dashboard
```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

**Dashboard Features:**
- Upload traffic video files (mp4, avi, mov, webm)
- Adjust confidence threshold with slider
- View real-time detection results
- Monitor email alert status
- See accident detection alerts

### Run Standalone Notebook
```bash
jupyter notebook accident_detection.ipynb
```

Processes video/webcam feed and saves output to `output_video/` folder.

## 📁 Project Structure

```
├── app.py                          # Main Streamlit web application
├── accident_detection.ipynb        # Standalone Jupyter notebook
├── requirements.txt                # Python dependencies
├── best.pt                         # Accident detection model
├── emergency.pt                    # Emergency vehicle detection model
├── alarm.wav                       # Audio alert sound
│
├── Model Results and Details/
│   ├── data.yaml                   # Dataset configuration
│   ├── train.py                    # Model training script
│   ├── yolo12m.pt                  # YOLOv12 medium pretrained model
│   ├── yolo11n.pt                  # YOLOv11 nano pretrained model
│   │
│   ├── Accident model results/     # Accident model training results
│   │   ├── args.yaml
│   │   ├── results.csv
│   │   └── [confusion matrices, PR curves, etc.]
│   │
│   └── Vehicle Classification/     # Emergency vehicle model results
│       ├── args.yaml
│       ├── results.csv
│       ├── weights/
│       │   ├── best.pt
│       │   └── last.pt
│       └── [training visualizations]
│
├── Testing Videos/                 # Sample video files
│   ├── rear_collision_10.mp4
│   ├── stock-footage-a-uk-ambulance...webm
│   └── WhatsApp Video 2025-12-05...mp4
│
└── output_video/                   # Generated video outputs
```

## 🤖 Models

### Emergency Vehicle Detection Model
- **File:** `emergency.pt`
- **Framework:** YOLOv12
- **Classes:** Ambulance, Fire Truck, Police, Vehicle
- **Dataset:** Indian Emergency Vehicles (Roboflow)
- **License:** CC BY 4.0

### Accident Detection Model
- **File:** `best.pt`
- **Framework:** YOLO
- **Purpose:** Real-time accident/collision detection
- **Training Data:** Custom accident dataset

## 🔧 Training (Advanced)

To retrain models with new data:

```bash
python Model\ Results\ and\ Details\train.py
```

**Training Configuration:**
- Epochs: 120
- Batch Size: 8 (RTX 3050 compatible)
- Learning Rate: 0.001
- Optimizer: AdamW
- Image Size: 640x640

## 📊 Model Performance

Both models include comprehensive evaluation metrics:
- Confusion matrices
- Precision-Recall (PR) curves
- F1, Precision, and Recall curves
- Training/validation visualizations
- Results CSV files

## 🐛 Troubleshooting

### Streamlit Won't Start
- **Issue:** Exit code 1
- **Solution:** Set environment variables for email credentials
  ```bash
  $env:SENDER_EMAIL = "email@gmail.com"
  $env:SENDER_PASSWORD = "password"
  $env:RECEIVER_EMAIL = "recipient@gmail.com"
  ```

### GPU Not Detected
- Ensure NVIDIA CUDA Toolkit is installed
- Update NVIDIA drivers
- Verify PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`

### Model Not Found
- Ensure `best.pt` and `emergency.pt` are in the project root directory
- Check file paths are correct

### Email Alerts Not Working
- Verify environment variables are set correctly
- Use App Password for Gmail (not regular password)
- Ensure 5-minute cooldown between alerts
- Check SENDER_EMAIL and RECEIVER_EMAIL are valid

## 📦 Dependencies

See [requirements.txt](requirements.txt) for complete list:
- streamlit - Web UI framework
- opencv-python - Video processing
- ultralytics - YOLO models
- pygame - Audio alerts
- torch - Deep learning framework
- numpy - Numerical operations
- Pillow - Image processing

## 📝 Notes

- Email alerts have a 5-minute cooldown to prevent spam
- Confidence threshold default: 0.3 (adjustable in sidebar)
- GPS location requires browser permission
- Accident frame screenshots are automatically saved and deleted after email

## 🎯 Future Enhancements

- Real-time traffic flow analysis
- Traffic congestion prediction
- Multi-camera support
- Database logging for incident history
- Mobile app integration
- SMS alerts
- Integration with traffic management systems

## 📧 Support

For issues or questions, refer to:
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Roboflow Dataset](https://universe.roboflow.com/emergency-fksgw/indian-emergency-vehicles-pm1gh)

## 📄 License

Dataset License: CC BY 4.0
Project: For educational and commercial use.

---

**Version:** 1.0  
**Last Updated:** December 2025
