# Advanced Driver Monitoring System

A comprehensive deep learning-based system for monitoring driver behavior and safety in real-time. The system uses state-of-the-art computer vision and deep learning techniques to detect various aspects of driver behavior.

## Features

### Face Detection and Landmark Tracking
- Uses MediaPipe Face Mesh for accurate facial landmark detection
- Real-time face tracking and landmark visualization
- Robust to different lighting conditions and head poses

### Drowsiness Detection
- Hybrid approach combining:
  - Eye Aspect Ratio (EAR) calculation
  - Deep learning-based eye state classification
  - Temporal analysis for reliable drowsiness detection
- Custom CNN model for eye state classification
- Real-time alerts for drowsy behavior

### Head Pose Estimation
- Advanced head pose tracking using facial landmarks
- Detection of head orientation (left, right, up, down)
- Visual indicators for head position
- Distraction monitoring and alerts

### Phone Usage Detection
- Combined approach using:
  - Hand tracking with MediaPipe Hands
  - Object detection using ResNet50
  - Spatial relationship analysis between hands and face
- Real-time phone usage alerts
- Hand landmark visualization

### Seatbelt Detection
- Dual-method detection:
  - Deep learning-based classification using custom CNN
  - Advanced image processing with adaptive thresholding
  - Line detection and filtering
- Real-time seatbelt status monitoring
- Visual indicators for seatbelt detection

### Statistics and Monitoring
- Real-time statistics display
- Event counting and logging
- Alert system with cooldown
- Semi-transparent overlay for information display

## Requirements

- Python 3.7+
- CUDA-capable GPU (recommended)
- Webcam
- Required packages (listed in requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd driver-monitoring-system
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download required model files:
- MediaPipe will download required models automatically
- For custom models, they will be downloaded on first run

## Usage

1. Run the main script:
```bash
python src/driver_monitoring_system.py
```

2. System features:
- Press 'q' to quit the application
- Real-time alerts will be displayed on screen
- Statistics are shown in the top-right corner
- Alert logs are printed to the console

## Project Structure

```
src/
├── driver_monitoring_system.py  # Main system integration
├── models/
│   ├── face_detector.py        # Face detection and landmarks
│   ├── drowsiness_detector.py  # Drowsiness detection
│   ├── head_pose_detector.py   # Head pose estimation
│   ├── phone_detector.py       # Phone usage detection
│   └── seatbelt_detector.py    # Seatbelt detection
```

## Customization

You can adjust various parameters in the code:
- Detection thresholds
- Alert cooldown periods
- Model confidence thresholds
- UI elements and positions

## Notes

- Ensure good lighting conditions for optimal detection
- Camera should be positioned to clearly see the driver's face
- System performance depends on hardware capabilities
- Some features may require GPU for real-time performance

## Future Improvements

- Add data logging and analysis features
- Implement driver identification
- Add audio alerts
- Integrate with vehicle systems
- Add more behavioral monitoring features 