import cv2
import numpy as np
import time
from collections import deque
import os
import sys

class DriverMonitor:
    def __init__(self):
        # Load face and eye detectors
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Initialize counters and timers
        self.eye_counter = 0
        self.drowsy_start_time = None
        self.last_blink_time = None
        self.face_not_visible_start = None
        
        # Seatbelt detection parameters
        self.seatbelt_history = deque([False] * 5, maxlen=5)  # Track last 5 detections
        self.SEATBELT_MIN_ANGLE = 35  # Minimum angle for seatbelt
        self.SEATBELT_MAX_ANGLE = 55  # Maximum angle for seatbelt
        self.SEATBELT_MIN_LENGTH = 80  # Minimum length in pixels
        self.SEATBELT_MIN_AREA = 300   # Minimum contour area
        self.SEATBELT_MIN_RATIO = 3.0  # Minimum length/width ratio
        self.last_seatbelt_detection = None
        self.SEATBELT_ALERT_DURATION = 5.0  # Duration to show alert
        
        # Constants
        self.DROWSINESS_TIME = 15
        self.WARNING_TIME = 10
        self.BLINK_TIME_THRESHOLD = 0.5
        self.FACE_NOT_VISIBLE_TIME = 15.0
        
        # Statistics
        self.stats = {
            'drowsy_events': 0,
            'face_not_visible': 0,
            'face_not_visible_alerts': 0,
            'blink_count': 0,
            'seatbelt_detections': 0
        }
        
        # FPS calculation
        self.fps = 30
        self.fps_update_freq = 30
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        # Eye state tracking
        self.eye_state_history = deque([True] * 5, maxlen=5)
        self.last_eye_state = True
    
    def update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        if self.frame_count % self.fps_update_freq == 0:
            current_time = time.time()
            self.fps = self.fps_update_freq / (current_time - self.fps_start_time)
            self.fps_start_time = current_time
    
    def detect_eyes(self, gray, face):
        """Detect eyes in the face region with blink detection"""
        x, y, w, h = face
        
        # Define eye region (upper half of face)
        roi_gray = gray[y:y + h//2, x:x + w]
        
        # Detect eyes in the region
        eyes = self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Consider eyes closed if we can't detect both eyes
        eyes_open = len(eyes) >= 2
        current_time = time.time()
        
        # Handle state changes
        if eyes_open != self.last_eye_state:
            if eyes_open:  # Eyes just opened
                if self.last_blink_time is not None:
                    # Calculate blink duration
                    blink_duration = current_time - self.last_blink_time
                    if blink_duration <= self.BLINK_TIME_THRESHOLD:
                        # This was a normal blink
                        self.stats['blink_count'] += 1
                        self.drowsy_start_time = None  # Reset drowsiness counter
                    self.last_blink_time = None
            else:  # Eyes just closed
                self.last_blink_time = current_time
        
        self.last_eye_state = eyes_open
        
        # Update eye state history
        self.eye_state_history.append(eyes_open)
        
        # Return True if eyes are open or if it's just a normal blink
        if not eyes_open and self.last_blink_time is not None:
            blink_duration = current_time - self.last_blink_time
            return blink_duration <= self.BLINK_TIME_THRESHOLD
        
        return eyes_open
    
    def detect_seatbelt(self, frame, face):
        """Detect seat belt in the chest region with improved accuracy"""
        x, y, w, h = face
        frame_h, frame_w = frame.shape[:2]
        current_time = time.time()
        
        # Define chest region (below face, wider than face)
        chest_top = min(frame_h - 1, y + h)
        chest_height = int(h * 1.8)  # Increased height for better coverage
        chest_width = int(w * 2.5)   # Increased width for better coverage
        chest_left = max(0, x - w//2)
        
        # Ensure region stays within frame bounds
        chest_height = min(chest_height, frame_h - chest_top)
        chest_width = min(chest_width, frame_w - chest_left)
        
        if chest_height <= 0 or chest_width <= 0:
            return False
        
        # Extract chest region
        chest_region = frame[chest_top:chest_top + chest_height, 
                           chest_left:chest_left + chest_width]
        
        # Convert to grayscale
        gray = cv2.cvtColor(chest_region, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply bilateral filter for edge preservation
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Edge detection with multiple thresholds
        edges1 = cv2.Canny(filtered, 30, 90)
        edges2 = cv2.Canny(filtered, 20, 60)
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Create diagonal kernel for seatbelt orientation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        best_seatbelt = None
        max_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.SEATBELT_MIN_AREA:
                continue
            
            # Fit line to contour
            [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            angle = abs(np.degrees(np.arctan2(vy, vx)))
            
            # Get rotated rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            # Calculate length and width
            width = min(rect[1])
            length = max(rect[1])
            
            if width == 0 or length < self.SEATBELT_MIN_LENGTH:
                continue
            
            # Calculate aspect ratio
            ratio = length / width
            
            # Calculate straightness (how well the contour fits a line)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            straightness = area / hull_area if hull_area > 0 else 0
            
            # Score the detection based on multiple criteria
            angle_score = 1.0 - abs(angle - 45) / 45  # 45 degrees is ideal
            ratio_score = min(ratio / self.SEATBELT_MIN_RATIO, 1.0)
            straight_score = straightness
            
            # Combine scores
            score = angle_score * 0.4 + ratio_score * 0.3 + straight_score * 0.3
            
            if (self.SEATBELT_MIN_ANGLE <= angle <= self.SEATBELT_MAX_ANGLE and 
                ratio >= self.SEATBELT_MIN_RATIO and 
                straightness > 0.8 and
                score > max_score):
                max_score = score
                best_seatbelt = {
                    'box': box,
                    'score': score,
                    'angle': angle
                }
        
        # Update detection history and draw results
        seatbelt_detected = best_seatbelt is not None and max_score > 0.7
        self.seatbelt_history.append(seatbelt_detected)
        
        # Only consider it detected if majority of recent frames show detection
        consistent_detection = sum(self.seatbelt_history) >= 3
        
        if consistent_detection:
            if best_seatbelt:
                # Convert coordinates to global frame
                box_global = best_seatbelt['box'].astype(np.int32) + np.array([chest_left, chest_top], dtype=np.int32)
                
                # Draw seatbelt detection
                cv2.drawContours(frame, [box_global], 0, (0, 255, 0), 2)
                
                # Draw chest region
                cv2.rectangle(frame, 
                            (chest_left, chest_top),
                            (chest_left + chest_width, chest_top + chest_height),
                            (0, 255, 0), 2)
                
                # Draw detection info
                info_text = f"Seatbelt Detected (Score: {max_score:.2f}, Angle: {best_seatbelt['angle']:.1f}°)"
                cv2.putText(frame, info_text,
                          (chest_left, chest_top - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                self.stats['seatbelt_detections'] += 1
                self.last_seatbelt_detection = current_time
        else:
            # Draw warning if no seatbelt detected
            if current_time % 2 < 1:  # Flash warning every second
                cv2.putText(frame, "No Seatbelt Detected!",
                          (chest_left, chest_top - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw chest region in red
            cv2.rectangle(frame, 
                        (chest_left, chest_top),
                        (chest_left + chest_width, chest_top + chest_height),
                        (0, 0, 255), 2)
        
        return consistent_detection
    
    def process_frame(self, frame):
        """Process a single frame"""
        self.update_fps()
        current_time = time.time()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        if len(faces) == 0:
            # Start timing when face is not visible
            if self.face_not_visible_start is None:
                self.face_not_visible_start = current_time
            
            face_not_visible_duration = current_time - self.face_not_visible_start
            self.stats['face_not_visible'] += 1
            
            # Display warning message
            cv2.putText(frame, "Face Not Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display duration
            cv2.putText(frame, f"Face Missing: {face_not_visible_duration:.1f}s", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Alert if face has been missing for too long
            if face_not_visible_duration >= self.FACE_NOT_VISIBLE_TIME:
                cv2.putText(frame, "ATTENTION REQUIRED!", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1),
                            (0, 0, 255), 10)
                self.stats['face_not_visible_alerts'] += 1
            
            return frame
        else:
            self.face_not_visible_start = None
        
        # Process the largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Detect seatbelt
        seatbelt = self.detect_seatbelt(frame, face)
        
        # Check eyes
        eyes_open = self.detect_eyes(gray, face)
        
        if not eyes_open:
            if self.drowsy_start_time is None:
                self.drowsy_start_time = current_time
            
            drowsy_duration = current_time - self.drowsy_start_time
            
            # Only show drowsiness messages if it's not a normal blink
            if drowsy_duration > self.BLINK_TIME_THRESHOLD:
                # Display drowsiness duration
                cv2.putText(frame, f"Eyes Closed for: {drowsy_duration:.1f}s",
                           (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 0, 255), 2)
                
                # Warning phase
                if drowsy_duration > self.WARNING_TIME:
                    cv2.putText(frame, "DROWSINESS WARNING!",
                               (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, (0, 255, 255), 2)
                    cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1),
                                (0, 255, 255), 3)
                
                # Alert phase
                if drowsy_duration >= self.DROWSINESS_TIME:
                    self.stats['drowsy_events'] += 1
                    cv2.putText(frame, "DROWSINESS ALERT!",
                               (10, frame.shape[0] - 110), cv2.FONT_HERSHEY_SIMPLEX,
                               1.0, (0, 0, 255), 3)
                    cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1),
                                (0, 0, 255), 10)
        else:
            self.drowsy_start_time = None
            cv2.putText(frame, "Eyes Open", (10, frame.shape[0] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw statistics
        self.draw_statistics(frame)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}",
                   (frame.shape[1] - 120, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def draw_statistics(self, frame):
        """Draw monitoring statistics"""
        h, w = frame.shape[:2]
        stats_x = w - 200
        stats_y = 30
        line_height = 20
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (stats_x - 10, 10), (w - 10, stats_y + 130),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw statistics
        cv2.putText(frame, f"Drowsy Events: {self.stats['drowsy_events']}", 
                   (stats_x, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Blink Count: {self.stats['blink_count']}", 
                   (stats_x, stats_y + line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Face Not Visible: {self.stats['face_not_visible']}", 
                   (stats_x, stats_y + 2 * line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Face Alerts: {self.stats['face_not_visible_alerts']}", 
                   (stats_x, stats_y + 3 * line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Seatbelt Detections: {self.stats['seatbelt_detections']}", 
                   (stats_x, stats_y + 4 * line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    # Initialize video capture with error handling
    print("Initializing camera...")
    cap = None
    
    # Try different camera indices
    for index in [0, 1]:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Try DirectShow
        if cap is not None and cap.isOpened():
            break
        if cap is not None:
            cap.release()
    
    if cap is None or not cap.isOpened():
        print("Error: Could not open video capture device")
        print("Please make sure your webcam is connected and not in use by another application")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize driver monitor
    monitor = DriverMonitor()
    
    print("Driver Monitoring System Started")
    print("Press 'q' to quit")
    print("Drowsiness alert will trigger after 15 seconds of continuous eye closure")
    print("Warning will show after 10 seconds of eye closure")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                print("Attempting to reconnect...")
                cap.release()
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    print("Failed to reconnect. Please check your camera.")
                    break
                continue
            
            # Process frame
            processed_frame = monitor.process_frame(frame)
            
            # Display frame
            cv2.imshow('Driver Monitoring System', processed_frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    finally:
        # Cleanup
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 