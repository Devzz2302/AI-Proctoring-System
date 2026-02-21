from decouple import config

class Settings:
    PROJECT_NAME: str = "AI Proctoring System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = config("DATABASE_URL", default="sqlite:///./proctoring.db")
    
    # Model paths
    YOLO_MODEL_PATH: str = "yolov8n.pt"
    
    # Alert thresholds
    GAZE_THRESHOLD: float = 0.7
    PHONE_CONFIDENCE: float = 0.8
    MULTIPLE_FACE_CONFIDENCE: float = 0.7

settings = Settings()