# 🎥 CameraTriangulation: Stereo-Vision Playground 🕶️✨

Welcome to the ultimate stereo-vision toolkit that turns two camera feeds into 3D magic! Grab your chessboard, fire up your cameras and let’s dive into the world of real-time depth estimation.

## 🚀 Features
1. 🧩 **Chessboard calibration**  
   Compute sweet intrinsic parameters in a snap.  
2. 📸 **Synchronized dual-camera capture**  
   Capture perfectly aligned frames from two eyes.  
3. 🌈 **Lens undistortion**  
   Wave goodbye to fish-eye warping.  
4. 🤖 **YOLOv11 object detection**  
   Spot and label objects at lightning speed.  
5. 🎯 **Stereo triangulation**  
   Estimate real-world 3D positions of matched objects.  
6. 🖥️ **Flask-based PC server**  
   Host detection, distance calculation or simple frame relay.  
7. 🐧 **Raspberry Pi client**  
   Capture, undistort, encode & send frames over the network.

## 📂 Repository Structure
- `calibration.py` – Chessboard calibration → outputs `calibration_matrix1.yaml`, `calibration_matrix2.yaml`  
- `screenshots.py` – Snap synchronized stereo frames into `images1/` and `images2/`  
- `cameras_calibration_test.py` – Verify undistortion on live streams  
- `PC_yolo.py` – Flask server: only object detection  
- `PC_distance_calc.py` – Flask server: detection + 3D position estimation  
- `PC_cameras.py` – Flask server: simple frame relay  
- `raspberry.py` – Pi client: capture, undistort, encode & POST frames  
- `test/`  
  - `stereo_depth.py` – Depth map demos (BM & SGBM)  
  - `trangulation.py` – Pure-math triangulation playground  
  - `yolo_test.py` – SVD-based triangulation with YOLO

## ⚙️ Prerequisites
- Python 3.8+  
- `opencv-python`  
- `numpy`  
- `pyyaml`  
- `flask`  
- `requests`  
- `ultralytics` (YOLOv11)

### Install dependencies
```bash
pip install opencv-python numpy pyyaml flask requests ultralytics
```

## 🎮 Quick Start
1. **Calibrate** your cameras:  
   ```bash
   python calibration.py
   ```
2. **Capture** stereo snapshots:  
   ```bash
   python screenshots.py
   ```
3. **Verify** undistortion:  
   ```bash
   python cameras_calibration_test.py
   ```
4. **Run** YOLO server:  
   ```bash
   python PC_yolo.py
   ```
5. **Run** distance‐calc server:  
   ```bash
   python PC_distance_calc.py
   ```
6. **Launch** Raspberry Pi client:  
   ```bash
   python raspberry.py
   ```
   
---

Happy triangulating! 🎉