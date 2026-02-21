from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import base64
import json
from datetime import datetime
import os

app = FastAPI(title="AI Proctoring System")

# Serve static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service flags FIRST
FACE_AVAILABLE = False
OBJECT_AVAILABLE = False
ALERT_AVAILABLE = False

# Initialize detectors as None
face_detector = None
object_detector = None
alert_system = None

# FIXED: Simple face detection fallback
try:
    # Simple face detector using OpenCV
    class SimpleFaceDetector:
        def __init__(self):
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("✅ Simple Face detector initialized")
        
        def detect_faces(self, image_data):
            """Detect faces using OpenCV"""
            try:
                # Convert base64 to numpy array
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    return []
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
                
                # Convert to list of face data
                face_data = []
                for (x, y, w, h) in faces:
                    face_data.append({
                        'bbox': [x, y, x+w, y+h],
                        'confidence': 0.9
                    })
                
                return face_data
                
            except Exception as e:
                print(f"Face detection error: {e}")
                return []
        
        def draw_detections(self, frame, faces):
            """Draw face bounding boxes"""
            try:
                result_frame = frame.copy()
                for face in faces:
                    bbox = face['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(result_frame, "Face", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                return result_frame
            except Exception as e:
                print(f"Face drawing error: {e}")
                return frame
    
    face_detector = SimpleFaceDetector()
    FACE_AVAILABLE = True
    print("✅ Face detection loaded")
except Exception as e:
    print(f"❌ Face detection failed: {e}")
    FACE_AVAILABLE = False

try:
    # Use YOLO for object detection
    from ultralytics import YOLO
    
    class SimpleObjectDetector:
        def __init__(self):
            # Load YOLO model for object detection
            self.model = YOLO('yolov8n.pt')
            self.suspicious_classes = ['cell phone', 'laptop', 'book', 'bottle', 'person']
            print("✅ YOLO Object detector initialized")
        
        def detect_suspicious_objects(self, frame):
            """Detect objects using YOLO"""
            try:
                # Run inference
                results = self.model(frame)
                
                detections = []
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            class_id = int(box.cls[0])
                            class_name = self.model.names[class_id]
                            confidence = float(box.conf[0])
                            
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            # Only return suspicious objects
                            if class_name in self.suspicious_classes:
                                detections.append({
                                    'class': class_name,
                                    'confidence': confidence,
                                    'bbox': [x1, y1, x2, y2]
                                })
                
                return detections
                
            except Exception as e:
                print(f"Object detection error: {e}")
                return []
        
        def draw_detections(self, frame, objects):
            """Draw bounding boxes on frame"""
            try:
                result_frame = frame.copy()
                
                if not objects or not isinstance(objects, list):
                    return result_frame
                
                for obj in objects:
                    if not isinstance(obj, dict) or 'bbox' not in obj:
                        continue
                        
                    bbox = obj['bbox']
                    if len(bbox) != 4:
                        continue
                    
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Validate coordinates
                    height, width = result_frame.shape[:2]
                    x1 = max(0, min(x1, width))
                    x2 = max(0, min(x2, width))
                    y1 = max(0, min(y1, height))
                    y2 = max(0, min(y2, height))
                    
                    if x1 >= x2 or y1 >= y2:
                        continue
                    
                    # Draw rectangle (GREEN for objects)
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    class_name = obj.get('class', 'object')
                    confidence = obj.get('confidence', 0.0)
                    label = f"{class_name} {confidence:.2f}"
                    
                    label_y = max(15, y1 - 10)
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(result_frame, (x1, label_y - label_size[1] - 5), 
                                (x1 + label_size[0], label_y + 5), (0, 255, 0), -1)
                    cv2.putText(result_frame, label, (x1, label_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                return result_frame
                
            except Exception as e:
                print(f"Drawing error: {e}")
                return frame
            
    object_detector = SimpleObjectDetector()
    OBJECT_AVAILABLE = True
    print("✅ Object detection loaded")
except Exception as e:
    print(f"❌ Object detection failed: {e}")
    OBJECT_AVAILABLE = False

# FIXED: Simple alert system
try:
    class SimpleAlertSystem:
        def __init__(self):
            print("✅ Simple Alert system initialized")
        
        def evaluate_alert(self, detection_data):
            """Evaluate alerts based on detection data"""
            alerts = []
            
            # No face alert
            if detection_data['face_count'] == 0 and detection_data['no_face_duration'] > 5:
                alerts.append({
                    'type': 'no_face',
                    'severity': 'high',
                    'message': f"No face detected for {detection_data['no_face_duration']} seconds"
                })
            
            # Multiple faces alert
            if detection_data['face_count'] > 1:
                alerts.append({
                    'type': 'multiple_faces',
                    'severity': 'critical',
                    'message': f"Multiple faces detected: {detection_data['face_count']} people"
                })
            
            # Suspicious objects alert
            if detection_data['objects']:
                for obj in detection_data['objects']:
                    alerts.append({
                        'type': 'suspicious_object',
                        'severity': 'high',
                        'message': f"Suspicious object: {obj['class']} ({obj['confidence']:.2f})"
                    })
            
            return alerts
    
    alert_system = SimpleAlertSystem()
    ALERT_AVAILABLE = True
    print("✅ Alert system loaded")
except Exception as e:
    print(f"❌ Alert system failed: {e}")
    ALERT_AVAILABLE = False

# Session states
session_states = {}

@app.get("/")
def read_root():
    return {
        "message": "AI Proctoring System is running! 🚀",
        "status": "active",
        "services": {
            "face_detection": FACE_AVAILABLE,
            "object_detection": OBJECT_AVAILABLE,
            "alert_system": ALERT_AVAILABLE
        },
        "endpoints": {
            "test": "/test",
            "camera_test": "/test-camera", 
            "web_frontend": "/static/index.html"
        }
    }

@app.get("/test")
def test_endpoint():
    return {"status": "OK", "message": "Server is working!"}

@app.get("/test-camera")
def test_camera():
    """Test if camera works"""
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                return {
                    "camera_status": "working", 
                    "message": "✅ Camera is accessible",
                    "frame_size": f"{frame.shape[1]}x{frame.shape[0]}" if ret else "Unknown"
                }
            else:
                return {"camera_status": "error", "message": "❌ Could not read frame"}
        else:
            return {"camera_status": "error", "message": "❌ Camera not accessible"}
    except Exception as e:
        return {"camera_status": "error", "message": f"❌ Error: {str(e)}"}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    # Initialize session state
    session_states[session_id] = {
        'no_face_duration': 0,
        'alert_count': 0
    }
    
    print(f"✅ WebSocket connected: {session_id}")
    
    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message['type'] == 'frame':
                # Decode base64 image
                try:
                    frame_data = message['frame']
                    if ',' in frame_data:
                        frame_data = frame_data.split(',')[1]
                    
                    image_data = base64.b64decode(frame_data)
                    nparr = np.frombuffer(image_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        await websocket.send_json({
                            "type": "error", 
                            "message": "Failed to decode image"
                        })
                        continue
                    
                    # Perform detection
                    faces = []
                    objects = []
                    
                    if FACE_AVAILABLE and face_detector:
                        try:
                            faces = face_detector.detect_faces(image_data)
                        except Exception as e:
                            print(f"Face detection error: {e}")
                            faces = []
                    
                    if OBJECT_AVAILABLE and object_detector:
                        try:
                            objects = object_detector.detect_suspicious_objects(frame)
                        except Exception as e:
                            print(f"Object detection error: {e}")
                            objects = []
                    
                    # Update session state
                    state = session_states[session_id]
                    
                    # Face detection logic
                    if len(faces) == 0:
                        state['no_face_duration'] += 1
                    else:
                        state['no_face_duration'] = 0
                    
                    # Prepare detection data
                    detection_data = {
                        'face_count': len(faces),
                        'objects': objects,
                        'no_face_duration': state['no_face_duration'],
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Evaluate alerts
                    alerts = []
                    if ALERT_AVAILABLE and alert_system:
                        try:
                            alerts = alert_system.evaluate_alert(detection_data)
                            state['alert_count'] += len(alerts)
                        except Exception as e:
                            print(f"Alert system error: {e}")
                            alerts = []
                    
                    # Draw detections on frame
                    frame_with_detections = frame.copy()
                    
                    if FACE_AVAILABLE and face_detector:
                        try:
                            frame_with_detections = face_detector.draw_detections(frame_with_detections, faces)
                        except Exception as e:
                            print(f"Face drawing error: {e}")
                    
                    if OBJECT_AVAILABLE and object_detector:
                        try:
                            frame_with_detections = object_detector.draw_detections(frame_with_detections, objects)
                        except Exception as e:
                            print(f"Object drawing error: {e}")
                    
                    # Add stats to frame
                    try:
                        cv2.putText(frame_with_detections, f"Faces: {len(faces)}", 
                                   (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(frame_with_detections, f"Objects: {len(objects)}", 
                                   (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    except Exception as e:
                        print(f"Stats drawing error: {e}")
                    
                    # Convert frame back to base64
                    try:
                        _, buffer = cv2.imencode('.jpg', frame_with_detections, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        processed_frame = base64.b64encode(buffer).decode('utf-8')
                    except Exception as e:
                        print(f"Frame encoding error: {e}")
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        processed_frame = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send response
                    response = {
                        'type': 'analysis_result',
                        'detections': detection_data,
                        'alerts': alerts,
                        'processed_frame': f"data:image/jpeg;base64,{processed_frame}",
                        'session_stats': {
                            'total_alerts': state['alert_count'],
                            'no_face_time': state['no_face_duration']
                        }
                    }
                    
                    await websocket.send_json(response)
                    
                except Exception as e:
                    print(f"WebSocket processing error: {e}")
                    await websocket.send_json({
                        "type": "error", 
                        "message": f"Processing error: {str(e)}"
                    })
                    
    except WebSocketDisconnect:
        print(f"❌ WebSocket disconnected: {session_id}")
        if session_id in session_states:
            del session_states[session_id]

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Proctoring System...")
    print("📊 Available services:")
    print(f"   - Face Detection: {'✅' if FACE_AVAILABLE else '❌'}")
    print(f"   - Object Detection: {'✅' if OBJECT_AVAILABLE else '❌'}")
    print(f"   - Alert System: {'✅' if ALERT_AVAILABLE else '❌'}")
    print("🌐 Open http://localhost:8000/static/index.html in your browser")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)