import os
import cv2
import numpy as np
import shutil
import yaml
from ultralytics import YOLO
import json
import random

class DatasetProcessor:
    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path
        self.classes = ['cheating', 'normal']
        self.yolo_data = {}
        
    def analyze_dataset_structure(self):
        """Analyze the OEP dataset structure"""
        print("🔍 Analyzing dataset structure...")
        
        if not os.path.exists(self.dataset_path):
            print(f"❌ Dataset path {self.dataset_path} not found!")
            return False
        
        print(f"📁 Dataset location: {os.path.abspath(self.dataset_path)}")
        print("📂 Folder contents:")
        
        # Count video files
        video_files = self._find_video_files()
        print(f"🎥 Found {len(video_files)} video files")
        
        # Check for ground truth files
        gt_files = self._find_gt_files()
        print(f"📝 Found {len(gt_files)} ground truth files")
        
        if len(video_files) > 0:
            print("✅ Valid dataset structure detected (video-based)")
            return True
        else:
            print("⚠ No video files found")
            return False
    
    def _find_video_files(self):
        """Find all video files in the dataset"""
        video_extensions = ['.avi', '.mp4', '.mov', '.mkv']
        video_files = []
        
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(os.path.join(root, file))
        
        return video_files
    
    def _find_gt_files(self):
        """Find ground truth files"""
        gt_files = []
        
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.lower() == 'gt.txt':
                    gt_files.append(os.path.join(root, file))
        
        return gt_files
    
    def extract_frames_from_videos(self):
        """Extract frames from video files for training"""
        print("🎬 Extracting frames from videos...")
        
        video_files = self._find_video_files()
        if not video_files:
            print("❌ No video files found to extract frames from")
            return False
        
        frames_dir = os.path.join(self.dataset_path, 'extracted_frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        frame_count = 0
        for video_path in video_files[:5]:  # Process first 5 videos for demo
            try:
                cap = cv2.VideoCapture(video_path)
                video_frames = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Extract every 10th frame to avoid too many similar frames
                    if video_frames % 10 == 0:
                        frame_filename = f"frame_{frame_count:06d}.jpg"
                        frame_path = os.path.join(frames_dir, frame_filename)
                        cv2.imwrite(frame_path, frame)
                        frame_count += 1
                    
                    video_frames += 1
                
                cap.release()
                print(f"✅ Extracted {video_frames//10} frames from {os.path.basename(video_path)}")
                
            except Exception as e:
                print(f"❌ Error processing {video_path}: {e}")
        
        print(f"🎯 Total frames extracted: {frame_count}")
        return frame_count > 0
    
    def convert_to_yolo_format(self):
        """Convert OEP dataset to YOLO format"""
        print("🔄 Converting dataset to YOLO format...")
        
        # Create YOLO directory structure
        yolo_base = os.path.join(self.dataset_path, 'yolo_dataset')
        yolo_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
        for dir_path in yolo_dirs:
            os.makedirs(os.path.join(yolo_base, dir_path), exist_ok=True)
        
        # Try to extract frames from videos
        has_frames = self.extract_frames_from_videos()
        
        if has_frames:
            # Use extracted frames
            frames_dir = os.path.join(self.dataset_path, 'extracted_frames')
            frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
            
            if frame_files:
                print(f"✅ Using {len(frame_files)} extracted frames")
                self._create_dataset_from_frames(frame_files, frames_dir, yolo_base)
            else:
                print("⚠ No frames extracted, creating demo dataset")
                self._create_demo_dataset(yolo_base)
        else:
            print("⚠ Could not extract frames, creating demo dataset")
            self._create_demo_dataset(yolo_base)
        
        # Create dataset.yaml
        self._create_dataset_yaml(yolo_base)
        
        print("✅ Dataset conversion completed")
        return yolo_base
    
    def _create_dataset_from_frames(self, frame_files, frames_dir, yolo_base):
        """Create dataset from extracted frames"""
        # Split into train/val (80/20)
        random.shuffle(frame_files)
        split_index = int(0.8 * len(frame_files))
        train_files = frame_files[:split_index]
        val_files = frame_files[split_index:]
        
        # Copy training frames
        for i, frame_file in enumerate(train_files):
            src_path = os.path.join(frames_dir, frame_file)
            dest_path = os.path.join(yolo_base, 'images/train', frame_file)
            shutil.copy2(src_path, dest_path)
            
            # Create label file
            label_file = frame_file.replace('.jpg', '.txt')
            label_path = os.path.join(yolo_base, 'labels/train', label_file)
            self._create_label_from_gt(label_path, src_path)
        
        # Copy validation frames
        for i, frame_file in enumerate(val_files):
            src_path = os.path.join(frames_dir, frame_file)
            dest_path = os.path.join(yolo_base, 'images/val', frame_file)
            shutil.copy2(src_path, dest_path)
            
            # Create label file
            label_file = frame_file.replace('.jpg', '.txt')
            label_path = os.path.join(yolo_base, 'labels/val', label_file)
            self._create_label_from_gt(label_path, src_path)
        
        print(f"✅ Created dataset with {len(train_files)} training and {len(val_files)} validation images")
    
    def _create_label_from_gt(self, label_path, frame_path):
        """Create label file based on video context"""
        # For OEP dataset, we need to analyze the video context and gt.txt files
        # This is a simplified version - you'd parse the actual ground truth
        
        # Determine class based on filename patterns
        frame_name = os.path.basename(frame_path).lower()
        
        # Simple heuristic based on OEP dataset structure
        if '1.avi' in frame_path or '2.avi' in frame_path:
            # These might indicate different scenarios in OEP
            class_id = random.randint(0, 1)  # 0 or 1 for cheating/normal
        else:
            class_id = 2  # normal behavior
        
        # Create annotation (center x, center y, width, height)
        with open(label_path, 'w') as f:
            f.write(f"{class_id} 0.5 0.5 0.3 0.3\n")
    
    def _create_demo_dataset(self, yolo_base):
        """Create demo dataset with proper images"""
        print("📸 Creating demo dataset with sample images...")
        
        # Create sample images using OpenCV (not empty files)
        for split in ['train', 'val']:
            for i in range(50):  # Create more samples
                img_path = os.path.join(yolo_base, 'images', split, f'image_{i:04d}.jpg')
                
                # Create a simple colored image
                img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                cv2.imwrite(img_path, img)
                
                # Create label file
                label_path = os.path.join(yolo_base, 'labels', split, f'image_{i:04d}.txt')
                with open(label_path, 'w') as f:
                    class_id = i % 3
                    f.write(f"{class_id} 0.5 0.5 0.3 0.3\n")
    
    def _create_dataset_yaml(self, yolo_base):
        """Create dataset.yaml for YOLO training"""
        dataset_config = {
            'path': os.path.abspath(yolo_base),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 3,
            'names': {
                0: 'phone_usage',
                1: 'gaze_aversion', 
                2: 'normal_behavior'
            }
        }
        
        # Save dataset.yaml
        yaml_path = os.path.join(yolo_base, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f)
        
        self.yolo_data = yaml_path  # Store path instead of dict
        print(f"✅ Dataset config created at: {yaml_path}")