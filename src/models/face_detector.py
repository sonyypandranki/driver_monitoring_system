import mediapipe as mp
import cv2
import numpy as np

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def detect_face(self, frame):
        """Detect face and return face landmarks"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection_result = self.face_detection.process(frame_rgb)
        mesh_result = self.face_mesh.process(frame_rgb)
        
        face_data = {
            'face_detected': False,
            'landmarks': None,
            'bbox': None
        }
        
        if detection_result.detections and mesh_result.multi_face_landmarks:
            face_data['face_detected'] = True
            face_data['landmarks'] = mesh_result.multi_face_landmarks[0]
            
            detection = detection_result.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            bbox = {
                'xmin': int(bbox.xmin * w),
                'ymin': int(bbox.ymin * h),
                'width': int(bbox.width * w),
                'height': int(bbox.height * h)
            }
            face_data['bbox'] = bbox
            
        return face_data
    
    def draw_face_landmarks(self, frame, landmarks):
        """Draw facial landmarks on the frame"""
        if landmarks:
            h, w, _ = frame.shape
            for landmark in landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        return frame 