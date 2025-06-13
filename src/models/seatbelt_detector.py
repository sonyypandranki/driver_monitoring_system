import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation

class SeatbeltDetector:
    def __init__(self):
        # Initialize the seatbelt detection model
        self.model = SeatbeltNet()
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Detection parameters
        self.confidence_threshold = 0.7
        self.no_seatbelt_frames = 0
        self.FRAMES_THRESHOLD = 20
        
        # Line detection parameters
        self.min_line_length = 100
        self.max_line_gap = 10
        self.angle_threshold = 15
        
    def preprocess_frame(self, frame):
        """Preprocess frame for the neural network"""
        # Resize frame
        frame_resized = cv2.resize(frame, (224, 224))
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        # Convert to tensor and normalize
        frame_tensor = self.transform(frame_rgb)
        return frame_tensor.unsqueeze(0)
    
    def detect_lines(self, frame):
        """Detect lines in the frame using advanced image processing"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Edge detection
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
        
        # Apply morphological operations
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # Detect lines using probabilistic Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                               minLineLength=self.min_line_length,
                               maxLineGap=self.max_line_gap)
        
        return lines
    
    def filter_seatbelt_lines(self, lines, frame_shape):
        """Filter lines to identify potential seatbelt lines"""
        if lines is None:
            return []
        
        seatbelt_lines = []
        h, w = frame_shape[:2]
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line angle
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Filter lines based on angle and position
            if (30 <= angle <= 60) or (120 <= angle <= 150):
                # Check if line is in the expected region (upper body area)
                if (y1 > h * 0.2 and y1 < h * 0.8 and
                    y2 > h * 0.2 and y2 < h * 0.8):
                    seatbelt_lines.append(line[0])
        
        return seatbelt_lines
    
    def detect_seatbelt(self, frame):
        """Detect seatbelt using both deep learning and image processing"""
        # Deep learning detection
        input_tensor = self.preprocess_frame(frame)
        
        with torch.no_grad():
            prediction = self.model(input_tensor)
            confidence = torch.sigmoid(prediction).item()
        
        # Image processing detection
        lines = self.detect_lines(frame)
        seatbelt_lines = self.filter_seatbelt_lines(lines, frame.shape)
        
        # Combine both detections
        seatbelt_detected = (confidence > self.confidence_threshold and
                           len(seatbelt_lines) > 0)
        
        # Update counter
        if not seatbelt_detected:
            self.no_seatbelt_frames += 1
        else:
            self.no_seatbelt_frames = 0
        
        # Draw detection results
        if self.no_seatbelt_frames >= self.FRAMES_THRESHOLD:
            cv2.putText(frame, "Please Wear Seatbelt!", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw detected seatbelt lines
        for line in seatbelt_lines:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return self.no_seatbelt_frames >= self.FRAMES_THRESHOLD, frame

class SeatbeltNet(nn.Module):
    def __init__(self):
        super(SeatbeltNet, self).__init__()
        # CNN layers for seatbelt detection
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 14 * 14, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x 