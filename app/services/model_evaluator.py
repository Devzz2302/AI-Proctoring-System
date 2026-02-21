from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
import json

class ModelEvaluator:
    def __init__(self):
        self.metrics_history = []
    
    def evaluate_model(self, model, dataset_config):
        """Comprehensive model evaluation"""
        print("📊 Evaluating model performance...")
        
        # Validation metrics
        metrics = model.val(
            data=dataset_config['path'],
            split='val',
            conf=0.001,
            iou=0.6,
            device='cpu'
        )
        
        # Extract key metrics
        evaluation_results = {
            'map50': metrics.box.map50,
            'map': metrics.box.map,
            'precision': metrics.box.p,
            'recall': metrics.box.r,
            'f1': 2 * metrics.box.p * metrics.box.r / (metrics.box.p + metrics.box.r + 1e-16)
        }
        
        # Generate plots
        self._generate_evaluation_plots(metrics, model)
        
        # Save metrics
        self.metrics_history.append(evaluation_results)
        self._save_metrics(evaluation_results)
        
        return evaluation_results
    
    def _generate_evaluation_plots(self, metrics, model):
        """Generate evaluation plots and charts"""
        try:
            # Confusion matrix
            if hasattr(metrics, 'confusion_matrix'):
                plt.figure(figsize=(10, 8))
                sns.heatmap(metrics.confusion_matrix.matrix, 
                           annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.savefig('evaluation_results/confusion_matrix.png')
                plt.close()
            
            # Precision-Recall curve
            plt.figure(figsize=(10, 6))
            # This would require access to prediction probabilities
            plt.title('Precision-Recall Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.grid(True)
            plt.savefig('evaluation_results/precision_recall_curve.png')
            plt.close()
            
            # Training history (if available)
            if hasattr(model, 'trainer') and hasattr(model.trainer, 'metrics'):
                self._plot_training_history(model.trainer.metrics)
                
        except Exception as e:
            print(f"⚠ Could not generate plots: {e}")
    
    def _plot_training_history(self, metrics):
        """Plot training history"""
        plt.figure(figsize=(12, 8))
        
        # Plot loss curves
        if 'train/box_loss' in metrics:
            plt.subplot(2, 2, 1)
            plt.plot(metrics['train/box_loss'], label='Train Box Loss')
            plt.plot(metrics['val/box_loss'], label='Val Box Loss')
            plt.title('Box Loss')
            plt.legend()
        
        # Plot accuracy metrics
        if 'metrics/mAP50' in metrics:
            plt.subplot(2, 2, 2)
            plt.plot(metrics['metrics/mAP50'], label='mAP50')
            plt.title('mAP50')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('evaluation_results/training_history.png')
        plt.close()
    
    def _save_metrics(self, metrics):
        """Save evaluation metrics"""
        os.makedirs('evaluation_results', exist_ok=True)
        
        with open('evaluation_results/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create a readable report
        report = f"""
        AI Proctoring Model Evaluation Report
        =====================================
        
        Performance Metrics:
        - mAP50: {metrics.get('map50', 0):.4f}
        - mAP50-95: {metrics.get('map', 0):.4f} 
        - Precision: {metrics.get('precision', 0):.4f}
        - Recall: {metrics.get('recall', 0):.4f}
        - F1 Score: {metrics.get('f1', 0):.4f}
        
        Model Architecture:
        - Custom YOLOv8 with modified C2f and SPPF layers
        - Enhanced feature extraction for proctoring scenarios
        - Optimized for cheating behavior detection
        """
        
        with open('evaluation_results/report.txt', 'w') as f:
            f.write(report)