# 🎯 Object Detection & Tracking (YOLOv8)

A web application built with Streamlit and YOLOv8 for real-time object detection and tracking. It supports both Image and Video uploads, assigning unique tracking IDs to objects.

## ✨ Features
- **Image Detection**: Upload any image to detect objects.
- **Video Tracking**: Upload a video to track objects across frames with unique IDs (using ByteTrack).
- **Live Camera Script**: Includes a separate script (`live_camera.py`) for real-time webcam detection.

## 🛠️ Tech Stack
- Python 3.10+
- Streamlit (Web Interface)
- YOLOv8 (Ultralytics)
- OpenCV
- ByteTrack (Tracking Algorithm)

## 🚀 How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/saniyawaseem009/CodeAlpha_ObjectDetection.git
   ```
   
2. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

3. Run the web app:
   ```bash 
   streamlit run app.py
   ```

4. (Optional) Run the live camera script:
   ```bash
   python3 live_camera.py
   ```
