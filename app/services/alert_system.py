from datetime import datetime

class AlertSystem:
    def __init__(self):
        self.alert_thresholds = {
            'multiple_faces': 1,
            'phone_detected': 0.6,  # Lower confidence for phones
            'no_face': 3,
            'suspicious_object': 0.5
        }
        self.alert_history = []
        print("✅ Enhanced Alert system initialized")
    
    def evaluate_alert(self, detection_data: dict) -> list:
        """Evaluate detection data and generate alerts"""
        alerts = []
        
        # Phone detection (HIGH PRIORITY)
        phone_detections = [obj for obj in detection_data.get('objects', []) 
                           if obj['label'] == 'phone_usage']
        
        if phone_detections:
            best_phone = max(phone_detections, key=lambda x: x['confidence'])
            if best_phone['confidence'] > self.alert_thresholds['phone_detected']:
                alerts.append({
                    'type': 'phone_detected',
                    'severity': 'critical',
                    'message': f"📱 MOBILE PHONE DETECTED! Confidence: {best_phone['confidence']:.2f}",
                    'timestamp': datetime.now().isoformat(),
                    'confidence': best_phone['confidence']
                })
        
        # Multiple faces detection
        face_count = detection_data.get('face_count', 0)
        if face_count > self.alert_thresholds['multiple_faces']:
            alerts.append({
                'type': 'multiple_faces',
                'severity': 'high',
                'message': f"👥 MULTIPLE PEOPLE DETECTED: {face_count} persons",
                'timestamp': datetime.now().isoformat()
            })
        
        # No face detected
        no_face_duration = detection_data.get('no_face_duration', 0)
        if no_face_duration > self.alert_thresholds['no_face']:
            alerts.append({
                'type': 'no_face',
                'severity': 'high',
                'message': f"👤 NO FACE DETECTED for {no_face_duration} seconds",
                'timestamp': datetime.now().isoformat()
            })
        
        # Other suspicious objects
        other_objects = [obj for obj in detection_data.get('objects', []) 
                        if obj['label'] in ['suspicious_material', 'secondary_device']]
        
        for obj in other_objects:
            if obj['confidence'] > self.alert_thresholds['suspicious_object']:
                alerts.append({
                    'type': 'suspicious_object',
                    'severity': 'medium',
                    'message': f"⚠ {obj['label'].replace('_', ' ').upper()} detected",
                    'timestamp': datetime.now().isoformat()
                })
        
        # Log alerts
        for alert in alerts:
            self.alert_history.append(alert)
            print(f"🚨 {alert['severity'].upper()} ALERT: {alert['message']}")
        
        return alerts
    
    def get_recent_alerts(self, limit: int = 10) -> list:
        """Get recent alerts"""
        return self.alert_history[-limit:]