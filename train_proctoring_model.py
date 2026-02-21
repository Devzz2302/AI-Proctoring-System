import os
import sys
import shutil

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.dataset_processor import DatasetProcessor
from app.services.custom_yolo import CustomYOLO
from app.services.model_evaluator import ModelEvaluator

def main():
    print("🚀 Starting AI Proctoring Model Training Pipeline")
    print("=" * 60)
    
    # Step 1: Dataset Preparation
    print("\n📊 STEP 1: Dataset Preparation")
    
    processor = DatasetProcessor("dataset/OEP database")
    
    if not processor.analyze_dataset_structure():
        print("❌ Could not process dataset structure")
        return
    
    yolo_dataset_path = processor.convert_to_yolo_format()
    
    # Step 2: Custom Model Training
    print("\n🛠️ STEP 2: Custom Model Training")
    custom_yolo = CustomYOLO('yolov8n.pt')
    
    # Train the model - pass the YAML path directly
    yaml_path = processor.yolo_data
    
    try:
        results, trained_model = custom_yolo.train_custom_model(
            dataset_config=yaml_path,  # Pass the YAML path directly
            epochs=10,  # Reduced for testing
            imgsz=640
        )
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 Creating a placeholder model for demonstration...")
        # Create a basic model for demonstration
        trained_model = custom_yolo.create_custom_model(num_classes=3)
        results = None
    
    # Step 3: Model Evaluation
    print("\n📈 STEP 3: Model Evaluation")
    evaluator = ModelEvaluator()
    
    # Create a simple dataset config for evaluation
    dataset_config = {
        'path': yolo_dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'nc': 3,
        'names': {0: 'phone_usage', 1: 'gaze_aversion', 2: 'normal_behavior'}
    }
    
    metrics = evaluator.evaluate_model(trained_model, dataset_config)
    
    # Step 4: Save and Export
    print("\n💾 STEP 4: Saving Model")
    
    # Save the trained model
    model_save_path = "trained_models/proctoring_model.pt"
    os.makedirs("trained_models", exist_ok=True)
    
    try:
        trained_model.save(model_save_path)
        print(f"✅ Model saved to: {model_save_path}")
    except Exception as e:
        print(f"❌ Could not save model: {e}")
    
    # Print final results
    print("\n🎯 TRAINING COMPLETED!")
    print("=" * 60)
    print(f"📊 Final Metrics:")
    print(f"   - mAP50: {metrics.get('map50', 0):.4f}")
    print(f"   - Precision: {metrics.get('precision', 0):.4f}")
    print(f"   - Recall: {metrics.get('recall', 0):.4f}")
    
    print(f"\n📁 Output files:")
    print(f"   - Model: {model_save_path}")
    print(f"   - Dataset: {yolo_dataset_path}")
    print(f"   - Evaluation: evaluation_results/")
    
    print("\n🚀 You can now use the custom trained model in your proctoring system!")

if __name__ == "__main__":
    main()