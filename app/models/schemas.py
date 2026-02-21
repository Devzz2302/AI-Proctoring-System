from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, Text
from sqlalchemy.sql import func
from app.db.database import Base

class ExamSession(Base):
    __tablename__ = "exam_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String, index=True)
    exam_id = Column(String, index=True)
    start_time = Column(DateTime, default=func.now())
    end_time = Column(DateTime, nullable=True)
    status = Column(String, default="active")

class SuspiciousActivity(Base):
    __tablename__ = "suspicious_activities"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, index=True)
    activity_type = Column(String)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=func.now())
    frame_data = Column(Text)
    resolved = Column(Boolean, default=False)

# Create tables
Base.metadata.create_all(bind=engine)