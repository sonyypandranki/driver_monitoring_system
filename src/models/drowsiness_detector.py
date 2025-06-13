import torch
import torch.nn as nn
import numpy as np
import cv2
from scipy.spatial import distance

class DrowsinessDetector:
    def __init__(self):
        self.eye_aspect_ratio_threshold = 0.25
        self.drowsy_frames_threshold = 15
        self.drowsy_frames_counter = 0
        
        # Eye landmarks indices for MediaPipe Face Mesh
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # Initialize CNN model for eye state classification
        self.eye_state_model = EyeStateClassifier()
        
    def calculate_ear(self, landmarks, eye_indices):
        """Calculate eye aspect ratio"""
        points = []
        for idx in eye_indices:
            point = landmarks.landmark[idx]
            points.append([point.x, point.y])
        points = np.array(points)
        
        # Calculate vertical distances
        A = distance.euclidean(points[1], points[5])
        B = distance.euclidean(points[2], points[4])
        # Calculate horizontal distance
        C = distance.euclidean(points[0], points[3])
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C)
        return ear
    
    def extract_eye_region(self, frame, landmarks, eye_indices):
        """Extract eye region from frame using landmarks"""
        h, w, _ = frame.shape
        points = np.array([[landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h] 
                          for idx in eye_indices], dtype=np.int32)
        
        # Get bounding box of eye region
        x, y, w, h = cv2.boundingRect(points)
        eye_region = frame[y:y+h, x:x+w]
        
        if eye_region.size > 0:
            eye_region = cv2.resize(eye_region, (64, 32))
            return eye_region
        return None
    
    def detect_drowsiness(self, frame, landmarks):
        """Detect drowsiness using both EAR and CNN"""
        if not landmarks:
            return False, frame
        
        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(landmarks, self.LEFT_EYE)
        right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Extract eye regions
        left_eye = self.extract_eye_region(frame, landmarks, self.LEFT_EYE)
        right_eye = self.extract_eye_region(frame, landmarks, self.RIGHT_EYE)
        
        is_drowsy = False
        if avg_ear < self.eye_aspect_ratio_threshold:
            if left_eye is not None and right_eye is not None:
                # Use CNN to confirm eye state
                left_state = self.eye_state_model.predict(left_eye)
                right_state = self.eye_state_model.predict(right_eye)
                
                if left_state == 'closed' and right_state == 'closed':
                    self.drowsy_frames_counter += 1
                    if self.drowsy_frames_counter >= self.drowsy_frames_threshold:
                        is_drowsy = True
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            self.drowsy_frames_counter = 0
            
        return is_drowsy, frame

class EyeStateClassifier(nn.Module):
    def __init__(self):
        super(EyeStateClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 4, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 4)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def predict(self, eye_region):
        """Predict eye state (open/closed)"""
        if eye_region is None:
            return 'unknown'
        
        # Preprocess image
        eye_tensor = torch.from_numpy(eye_region).float().permute(2, 0, 1).unsqueeze(0)
        eye_tensor = eye_tensor / 255.0
        
        # Make prediction
        with torch.no_grad():
            output = self(eye_tensor)
            _, predicted = torch.max(output.data, 1)
            
        return 'closed' if predicted.item() == 1 else 'open' 