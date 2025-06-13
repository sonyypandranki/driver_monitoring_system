import cv2
import numpy as np
import dlib
import tensorflow as tf
from scipy.spatial import distance
from imutils import face_utils
import time

class DriverMonitoring:
    def __init__(self):
        # Initialize face detector and facial landmarks predictor
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        
        # Constants for eye aspect ratio
        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 20
        
        # Initialize counters
        self.drowsy_frames = 0
        self.phone_frames = 0
        self.no_seatbelt_frames = 0
        
        # Define facial landmarks for eyes
        self.LEFT_EYE_START = 36
        self.LEFT_EYE_END = 42
        self.RIGHT_EYE_START = 42
        self.RIGHT_EYE_END = 48

    def eye_aspect_ratio(self, eye):
        # Calculate euclidean distances between vertical eye landmarks
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        # Calculate euclidean distance between horizontal eye landmarks
        C = distance.euclidean(eye[0], eye[3])
        # Calculate eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear

    def check_drowsiness(self, frame, landmarks):
        # Extract eye coordinates
        left_eye = landmarks[self.LEFT_EYE_START:self.LEFT_EYE_END]
        right_eye = landmarks[self.RIGHT_EYE_START:self.RIGHT_EYE_END]
        
        # Calculate eye aspect ratios
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        if avg_ear < self.EYE_AR_THRESH:
            self.drowsy_frames += 1
            if self.drowsy_frames >= self.EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            self.drowsy_frames = 0

    def check_head_position(self, frame, landmarks):
        # Get nose coordinates as reference for head position
        nose = landmarks[27:36]
        nose_center = np.mean(nose, axis=0)
        
        # Define frame center
        frame_center = frame.shape[1] / 2
        
        # Check head position relative to frame center
        if nose_center[0] < frame_center - 100:
            cv2.putText(frame, "Head Position: Looking Left", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif nose_center[0] > frame_center + 100:
            cv2.putText(frame, "Head Position: Looking Right", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Head Position: Forward", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def check_phone_usage(self, frame, landmarks):
        # Check hand position near face using facial landmarks
        jaw = landmarks[0:17]
        jaw_center = np.mean(jaw, axis=0)
        
        # Simple color-based detection for potential phone presence
        roi = frame[int(jaw_center[1]):int(jaw_center[1]+100),
                   int(jaw_center[0]-50):int(jaw_center[0]+50)]
        
        if roi.size > 0:
            # Check for phone-like objects (simplified version)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (0, 0, 0), (180, 30, 255))
            
            if cv2.countNonZero(mask) > roi.size * 0.5:
                self.phone_frames += 1
                if self.phone_frames > 10:
                    cv2.putText(frame, "Phone Usage Detected!", (10, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.phone_frames = max(0, self.phone_frames - 1)

    def check_seatbelt(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Apply Hough transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=10)
        
        seatbelt_detected = False
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                # Check for diagonal lines typical of seatbelts
                if 30 < angle < 60:
                    seatbelt_detected = True
                    break
        
        if not seatbelt_detected:
            self.no_seatbelt_frames += 1
            if self.no_seatbelt_frames > 20:
                cv2.putText(frame, "Please Wear Seatbelt!", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            self.no_seatbelt_frames = 0

    def process_frame(self, frame):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_detector(gray, 0)
        
        for face in faces:
            # Get facial landmarks
            landmarks = self.landmark_predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)
            
            # Draw facial landmarks
            for (x, y) in landmarks:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
            # Perform all checks
            self.check_drowsiness(frame, landmarks)
            self.check_head_position(frame, landmarks)
            self.check_phone_usage(frame, landmarks)
            self.check_seatbelt(frame)
        
        return frame

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    driver_monitor = DriverMonitoring()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        processed_frame = driver_monitor.process_frame(frame)
        
        # Display the frame
        cv2.imshow('Driver Monitoring System', processed_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 