import numpy as np
import cv2
from scipy.spatial import distance

class HeadPoseDetector:
    def __init__(self):
        # Define MediaPipe face mesh indices for head pose estimation
        self.NOSE_TIP = 1
        self.NOSE_BRIDGE = 168
        self.LEFT_EYE = 33
        self.RIGHT_EYE = 263
        self.LEFT_MOUTH = 61
        self.RIGHT_MOUTH = 291
        
        # Thresholds for head pose classification
        self.YAW_THRESHOLD = 0.15  # Left-right rotation
        self.PITCH_THRESHOLD = 0.15  # Up-down rotation
        
    def get_face_orientation(self, landmarks):
        """Calculate face orientation using facial landmarks"""
        if not landmarks:
            return "unknown", 0, 0
        
        # Extract key points
        nose_tip = np.array([landmarks.landmark[self.NOSE_TIP].x,
                           landmarks.landmark[self.NOSE_TIP].y])
        nose_bridge = np.array([landmarks.landmark[self.NOSE_BRIDGE].x,
                              landmarks.landmark[self.NOSE_BRIDGE].y])
        left_eye = np.array([landmarks.landmark[self.LEFT_EYE].x,
                           landmarks.landmark[self.LEFT_EYE].y])
        right_eye = np.array([landmarks.landmark[self.RIGHT_EYE].x,
                            landmarks.landmark[self.RIGHT_EYE].y])
        left_mouth = np.array([landmarks.landmark[self.LEFT_MOUTH].x,
                             landmarks.landmark[self.LEFT_MOUTH].y])
        right_mouth = np.array([landmarks.landmark[self.RIGHT_MOUTH].x,
                              landmarks.landmark[self.RIGHT_MOUTH].y])
        
        # Calculate yaw (left-right rotation)
        eye_distance = distance.euclidean(left_eye, right_eye)
        mouth_distance = distance.euclidean(left_mouth, right_mouth)
        yaw = mouth_distance / eye_distance - 1
        
        # Calculate pitch (up-down rotation)
        vertical_distance = distance.euclidean(nose_tip, nose_bridge)
        pitch = vertical_distance / eye_distance - 0.5
        
        # Classify head pose
        if abs(yaw) < self.YAW_THRESHOLD and abs(pitch) < self.PITCH_THRESHOLD:
            orientation = "forward"
        elif yaw > self.YAW_THRESHOLD:
            orientation = "right"
        elif yaw < -self.YAW_THRESHOLD:
            orientation = "left"
        elif pitch > self.PITCH_THRESHOLD:
            orientation = "down"
        elif pitch < -self.PITCH_THRESHOLD:
            orientation = "up"
        else:
            orientation = "forward"
            
        return orientation, yaw, pitch
    
    def detect_head_pose(self, frame, landmarks):
        """Detect head pose and draw indicators on frame"""
        orientation, yaw, pitch = self.get_face_orientation(landmarks)
        
        # Draw head pose indicator
        if orientation != "unknown":
            text = f"Head Position: {orientation.capitalize()}"
            color = (0, 255, 0) if orientation == "forward" else (0, 0, 255)
            cv2.putText(frame, text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw direction arrow
            h, w, _ = frame.shape
            center_x = w // 2
            center_y = h // 2
            
            if orientation != "forward":
                arrow_length = 50
                if orientation == "left":
                    cv2.arrowedLine(frame, (center_x, center_y),
                                  (center_x - arrow_length, center_y),
                                  color, 2, tipLength=0.3)
                elif orientation == "right":
                    cv2.arrowedLine(frame, (center_x, center_y),
                                  (center_x + arrow_length, center_y),
                                  color, 2, tipLength=0.3)
                elif orientation == "up":
                    cv2.arrowedLine(frame, (center_x, center_y),
                                  (center_x, center_y - arrow_length),
                                  color, 2, tipLength=0.3)
                elif orientation == "down":
                    cv2.arrowedLine(frame, (center_x, center_y),
                                  (center_x, center_y + arrow_length),
                                  color, 2, tipLength=0.3)
        
        return orientation, frame 