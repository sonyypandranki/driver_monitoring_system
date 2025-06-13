import cv2
import numpy as np
import time
from models.face_detector import FaceDetector
from models.drowsiness_detector import DrowsinessDetector
from models.head_pose_detector import HeadPoseDetector
from models.phone_detector import PhoneDetector
from models.seatbelt_detector import SeatbeltDetector

class DriverMonitoringSystem:
    def __init__(self):
        # Initialize all detectors
        self.face_detector = FaceDetector()
        self.drowsiness_detector = DrowsinessDetector()
        self.head_pose_detector = HeadPoseDetector()
        self.phone_detector = PhoneDetector()
        self.seatbelt_detector = SeatbeltDetector()
        
        # Initialize state variables
        self.alert_active = False
        self.last_alert_time = 0
        self.alert_cooldown = 3.0  # seconds
        
        # Initialize statistics
        self.stats = {
            'drowsy_events': 0,
            'phone_usage_events': 0,
            'no_seatbelt_events': 0,
            'distracted_events': 0
        }
    
    def process_frame(self, frame):
        """Process a single frame and return the annotated frame"""
        # Detect face and get landmarks
        face_data = self.face_detector.detect_face(frame)
        
        if face_data['face_detected']:
            # Draw facial landmarks
            frame = self.face_detector.draw_face_landmarks(frame, face_data['landmarks'])
            
            # Check drowsiness
            is_drowsy, frame = self.drowsiness_detector.detect_drowsiness(frame, face_data['landmarks'])
            if is_drowsy:
                self.stats['drowsy_events'] += 1
                self._trigger_alert("DROWSINESS DETECTED!")
            
            # Check head pose
            head_orientation, frame = self.head_pose_detector.detect_head_pose(frame, face_data['landmarks'])
            if head_orientation != "forward":
                self.stats['distracted_events'] += 1
            
            # Check phone usage
            phone_detected, frame = self.phone_detector.detect_phone(frame, face_data['landmarks'])
            if phone_detected:
                self.stats['phone_usage_events'] += 1
                self._trigger_alert("PHONE USAGE DETECTED!")
        
        # Check seatbelt (independent of face detection)
        no_seatbelt, frame = self.seatbelt_detector.detect_seatbelt(frame)
        if no_seatbelt:
            self.stats['no_seatbelt_events'] += 1
            self._trigger_alert("PLEASE WEAR SEATBELT!")
        
        # Draw statistics
        self._draw_statistics(frame)
        
        return frame
    
    def _trigger_alert(self, message):
        """Trigger an alert if cooldown period has passed"""
        current_time = time.time()
        if current_time - self.last_alert_time >= self.alert_cooldown:
            self.alert_active = True
            self.last_alert_time = current_time
            
            # Log alert
            print(f"Alert: {message} - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _draw_statistics(self, frame):
        """Draw monitoring statistics on frame"""
        h, w = frame.shape[:2]
        stats_x = w - 200
        stats_y = 30
        line_height = 20
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (stats_x - 10, 10), (w - 10, stats_y + 90),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw statistics
        cv2.putText(frame, f"Drowsy Events: {self.stats['drowsy_events']}", 
                   (stats_x, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Phone Usage: {self.stats['phone_usage_events']}", 
                   (stats_x, stats_y + line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"No Seatbelt: {self.stats['no_seatbelt_events']}", 
                   (stats_x, stats_y + 2 * line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Distracted: {self.stats['distracted_events']}", 
                   (stats_x, stats_y + 3 * line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Initialize driver monitoring system
    monitoring_system = DriverMonitoringSystem()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame = monitoring_system.process_frame(frame)
        
        # Display frame
        cv2.imshow('Driver Monitoring System', processed_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 