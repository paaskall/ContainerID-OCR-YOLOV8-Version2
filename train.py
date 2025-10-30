from ultralytics import YOLO
import torch

print("🚀 Starting YOLOv8 Training with correct path...")

data_yaml_path = "ContainerID.v3i.yolov8/data.yaml"  
model = YOLO("container_id_training/yolov8_container_detectionV2/weights/last.pt")

# Training configuration
training_config = {
    'data': data_yaml_path,
    'epochs': 100,
    'imgsz': 640,
    'batch': 8,
    'patience': 15,
    'lr0': 0.01,
    'save': True,
    'save_period': 10,
    'device': 'cpu', 
    'workers': 2,
    'project': 'container_id_training',
    'name': 'yolov8_container_detectionV2',
    'exist_ok': True
}

print("Training Configuration:")
for k, v in training_config.items():
    print(f"   {k}: {v}")

# Start training
try:
    results = model.train(**training_config)
    print("Training completed!")
except Exception as e:
    print(f"Training error: {e}")
