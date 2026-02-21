from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict
import os

class CustomObjectDetector:
    def __init__(self, model_path=None):
        # Load pre-trained YOLO for object detection (phones, books, etc.)
        print("✅ Loading pre-trained YOLO for object detection...")
        self.object_model = YOLO('yolov8n.pt')  # Pre-trained for real objects
        
        # Also try to load custom model for proctoring behaviors
        self.custom_model = None
        self.custom_trained = False
        
        if model_path and os.path.exists(model_path):
            print(f"✅ Also loading custom trained model: {model_path}")
            self.custom_model = YOLO(model_path)
            self.custom_trained = True
        else:
            # Try to find the trained model automatically
            trained_model_path = "runs/detect/train/weights/best.pt"
            if os.path.exists(trained_model_path):
                print(f"✅ Also loading custom model: {trained_model_path}")
                self.custom_model = YOLO(trained_model_path)
                self.custom_trained = True
            else:
                print("⚠ Custom model not available, using pre-trained YOLO only")
        
        # Object mapping for pre-trained YOLO
        self.suspicious_objects = {
            'cell phone': 'phone_usage',
            'book': 'suspicious_material',
            'laptop': 'secondary_device',
            'person': 'multiple_persons'
        }
        
        print("✓ Enhanced Object Detector initialized")
        
    def detect_proctoring_activities(self, image: np.ndarray) -> List[Dict]:
        """Detect proctoring activities using both pre-trained and custom models"""
        detections = []
        
        try:
            # First, use pre-trained YOLO for real object detection
            object_results = self.object_model(image, verbose=False, conf=0.5)
            detections.extend(self._process_pretrained_detections(object_results))
            
            # Then, use custom model if available for proctoring behaviors
            if self.custom_model:
                custom_results = self.custom_model(image, verbose=False, conf=0.5)
                detections.extend(self._process_custom_detections(custom_results))
                
        except Exception as e:
            print(f"Object detection error: {e}")
        
        return detections
    
    def _process_pretrained_detections(self, results):
        """Process detections from pre-trained YOLO model"""
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = self.object_model.names[cls]
                    
                    # Only include suspicious objects
                    if label in self.suspicious_objects and conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detection = {
                            'label': self.suspicious_objects[label],
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2],
                            'class_id': cls,
                            'model_type': 'pre_trained',
                            'original_label': label
                        }
                        detections.append(detection)
        
        return detections
    
    def _process_custom_detections(self, results):
        """Process detections from custom trained model"""
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Map custom model classes
                    class_names = {
                        0: 'phone_usage',
                        1: 'gaze_aversion', 
                        2: 'normal_behavior'
                    }
                    
                    label = class_names.get(cls, f'class_{cls}')
                    
                    if conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detection = {
                            'label': label,
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2],
                            'class_id': cls,
                            'model_type': 'custom_trained'
                        }
                        detections.append(detection)
        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw object detection bounding boxes with custom colors"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            label = detection['label']
            conf = detection['confidence']
            model_type = detection.get('model_type', 'pre_trained')
            
            # Custom colors for different proctoring activities
            color_map = {
                'phone_usage': (0, 0, 255),      # Red
                'gaze_aversion': (0, 165, 255),  # Orange  
                'normal_behavior': (0, 255, 0),  # Green
                'multiple_persons': (255, 0, 0), # Blue
                'suspicious_material': (255, 255, 0), # Cyan
                'secondary_device': (255, 0, 255) # Magenta
            }
            
            color = color_map.get(label, (255, 0, 0))
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            # Create label text
            if model_type == 'custom_trained':
                label_text = f"CUSTOM: {label} ({conf:.2f})"
                text_color = (0, 255, 0)  # Green for custom model
            else:
                label_text = f"{label} ({conf:.2f})"
                text_color = color
            
            # Draw label background
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(image, label_text, 
                       (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            # Add model type indicator
            if model_type == 'custom_trained':
                cv2.putText(image, "CUSTOM AI MODEL", 
                           (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image