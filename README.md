# 🚗 Autonomous Driving Perception Engine


The Autonomous Driving Perception Engine processes road videos frame-by-frame and extracts meaningful environmental information required for autonomous navigation.

The pipeline detects and tracks dynamic road participants, estimates scene depth from monocular images, predicts future trajectories, identifies hazardous regions, and presents all information through an interactive driving intelligence dashboard.

This project is designed for:

- Autonomous Driving Research
- Advanced Driver Assistance Systems (ADAS)
- Intelligent Transportation Systems
- Robotics & Computer Vision
- AI Perception Studies

---

# ✨ Features

### 🚘 Vehicle Detection
Detects cars, buses, trucks, motorcycles, bicycles, and other road vehicles in real time.

### 🚶 Pedestrian Detection
Accurately identifies pedestrians and vulnerable road users.

### 🎯 Multi-Object Tracking
Maintains persistent IDs across video frames using object tracking.

### 🌍 Monocular Depth Estimation
Generates depth maps from a single RGB camera without requiring LiDAR.

### 🧊 3D Perception Visualization
Creates a pseudo-3D representation of detected objects using estimated depth.

### 📈 Future Path Prediction
Predicts probable trajectories of moving vehicles and pedestrians.

### 🛣️ Traffic Scene Understanding
Analyzes lanes, road users, traffic density, and surrounding context.

### ⚠️ Risk Zone Analysis
Highlights potential collision regions and unsafe driving situations.

### 📊 Driving Intelligence Dashboard
Displays detection statistics, tracking information, risk indicators, and perception outputs in an intuitive interface.

---

# 🏗️ System Architecture

```
Dashcam Video
      │
      ▼
Frame Extraction
      │
      ▼
Object Detection (YOLO)
      │
      ▼
Multi-Object Tracking
      │
      ▼
Depth Estimation
      │
      ▼
Trajectory Prediction
      │
      ▼
Risk Analysis
      │
      ▼
Scene Understanding
      │
      ▼
Visualization Dashboard
```

---

# 🧠 AI Components

- YOLOv8 Object Detection
- Multi-Object Tracking
- Monocular Depth Estimation
- Motion Analysis
- Future Trajectory Prediction
- Collision Risk Assessment
- Computer Vision Visualization

---

- Ultralytics YOLO
- OpenCV
- PyTorch
- NumPy
- Autonomous Driving Research Community
