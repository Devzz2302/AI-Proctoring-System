import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier()
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade.load(cascade_path)
    
    def detect_faces(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        detected_faces = []
        for (x, y, w, h) in faces:
            detected_faces.append({
                'xmin': x, 'ymin': y, 'xmax': x + w, 'ymax': y + h, 'confidence': 0.8
            })
        
        return detected_faces
    
    def draw_detections(self, image, faces):
        for face in faces:
            cv2.rectangle(image, (face['xmin'], face['ymin']), (face['xmax'], face['ymax']), (0, 255, 0), 2)
        return image