import os
import json
import shutil
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
import yaml
from datetime import datetime

class ModelTrainer:
    def __init__(self, enrolled_faces_dir="enrolled_faces", students_csv="students_list.csv"):
        self.enrolled_faces_dir = enrolled_faces_dir
        self.students_csv = students_csv
        self.yolo_dataset_dir = "yolo_dataset_updated"
        self.data_yaml_path = os.path.join(self.yolo_dataset_dir, "data.yaml")
        
    def prepare_yolo_dataset(self):
        """
        Convert enrolled student images to YOLO format
        """
        print("🔄 Preparing YOLO dataset from enrolled students...")
        
        # Create YOLO dataset structure
        for split in ['train', 'val']:
            for subdir in ['images', 'labels']:
                os.makedirs(os.path.join(self.yolo_dataset_dir, split, subdir), exist_ok=True)
        
        # Load students list
        if not os.path.exists(self.students_csv):
            raise FileNotFoundError(f"Students list file not found: {self.students_csv}")
        
        students_df = pd.read_csv(self.students_csv)
        
        # Create class mapping
        class_mapping = {}
        class_names = {}
        
        for idx, student_name in enumerate(students_df['student_name'].unique()):
            class_mapping[student_name] = idx
            class_names[idx] = student_name
        
        print(f"📋 Found {len(class_names)} students to train on")
        
        # Process each student's images
        image_count = 0
        for _, student_row in students_df.iterrows():
            student_name = student_row['student_name']
            student_dir = os.path.join(self.enrolled_faces_dir, student_name.replace(" ", "_"))
            
            if not os.path.exists(student_dir):
                print(f"⚠️ Warning: No images found for {student_name}")
                continue
            
            # Get all images for this student
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(Path(student_dir).glob(ext))
            
            if not image_files:
                print(f"⚠️ Warning: No valid images found for {student_name}")
                continue
            
            # Split images: 80% train, 20% val
            train_count = max(1, int(len(image_files) * 0.8))
            train_images = image_files[:train_count]
            val_images = image_files[train_count:]
            
            # Process training images
            for img_path in train_images:
                self._process_image(img_path, student_name, class_mapping, 'train')
                image_count += 1
            
            # Process validation images
            for img_path in val_images:
                self._process_image(img_path, student_name, class_mapping, 'val')
                image_count += 1
        
        # Create YAML configuration
        self._create_yaml_config(class_names)
        
        print(f"✅ Dataset prepared with {image_count} images and {len(class_names)} classes")
        return len(class_names), image_count
    
    def _process_image(self, img_path, student_name, class_mapping, split):
        """
        Process a single image and create corresponding label file
        """
        from PIL import Image
        
        # Copy image to YOLO dataset
        img_filename = f"{student_name.replace(' ', '_')}_{img_path.stem}.jpg"
        dst_img_path = os.path.join(self.yolo_dataset_dir, split, 'images', img_filename)
        
        # Open, resize if needed, and save image
        img = Image.open(img_path)
        img.save(dst_img_path)
        
        # Create label file (assuming full image is the face)
        label_filename = f"{student_name.replace(' ', '_')}_{img_path.stem}.txt"
        dst_label_path = os.path.join(self.yolo_dataset_dir, split, 'labels', label_filename)
        
        class_id = class_mapping[student_name]
        
        # For enrolled faces, assume the whole image contains the face
        # Center coordinates (0.5, 0.5) and full width/height (1.0, 1.0)
        with open(dst_label_path, 'w') as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
    
    def _create_yaml_config(self, class_names):
        """
        Create YAML configuration file for YOLO training
        """
        yaml_config = {
            'path': str(Path(self.yolo_dataset_dir).absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(class_names),
            'names': class_names
        }
        
        with open(self.data_yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"📄 YAML config created: {self.data_yaml_path}")
    
    def train_model(self, epochs=50, img_size=640, batch_size=16):
        """
        Train YOLOv8 model on the prepared dataset
        """
        print("🚀 Starting model training...")
        
        # Load YOLOv8n model
        model = YOLO('yolov8n.pt')
        
        # Train the model
        results = model.train(
            data=self.data_yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name='student_face_detection_updated',
            patience=10,
            save=True,
            plots=True,
            device=0 if self._is_gpu_available() else 'cpu',
            verbose=True,
            
            # Augmentation parameters
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
        )
        
        print("✅ Model training completed!")
        return results
    
    def _is_gpu_available(self):
        """
        Check if GPU is available for training
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def get_best_model_path(self):
        """
        Get the path to the best trained model
        """
        runs_dir = Path("runs/detect")
        if runs_dir.exists():
            # Find the latest student_face_detection_updated folder
            model_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('student_face_detection_updated')]
            if model_dirs:
                latest_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
                best_model_path = latest_dir / "weights" / "best.pt"
                if best_model_path.exists():
                    return str(best_model_path)
        return None
    
    def retrain_with_new_data(self, epochs=50, img_size=640, batch_size=16):
        """
        Complete retraining pipeline
        """
        try:
            # Step 1: Prepare dataset
            num_classes, num_images = self.prepare_yolo_dataset()
            
            if num_classes == 0:
                raise ValueError("No enrolled students found for training")
            
            # Step 2: Train model
            results = self.train_model(epochs, img_size, batch_size)
            
            # Step 3: Get best model path
            best_model_path = self.get_best_model_path()
            
            if best_model_path:
                print(f"✅ New model saved at: {best_model_path}")
                return best_model_path, results
            else:
                raise FileNotFoundError("Trained model not found")
                
        except Exception as e:
            print(f"❌ Error during retraining: {str(e)}")
            raise e

def main():
    """
    Example usage of the ModelTrainer
    """
    trainer = ModelTrainer()
    
    try:
        best_model_path, results = trainer.retrain_with_new_data(
            epochs=30,  # Reduced for testing
            img_size=640,
            batch_size=8
        )
        print(f"🎉 Retraining completed successfully!")
        print(f"📁 Best model: {best_model_path}")
        
    except Exception as e:
        print(f"💥 Retraining failed: {str(e)}")

if __name__ == "__main__":
    main()