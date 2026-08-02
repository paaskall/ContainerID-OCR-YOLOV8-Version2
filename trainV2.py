from ultralytics import YOLO
import os
import torch

print("Starting YOLOv8 Training (Auto Resume Mode)...")

data_yaml_path = "ContainerID.v3i.yolov8/data.yaml"
checkpoint_path = "container_id_training/yolov8_container_detectionV2/weights/last.pt"
base_model = "yolov8n.pt"

training_config = {
    'data': data_yaml_path,
    'epochs': 100,
    'imgsz': 640,
    'batch': 8,
    'patience': 15,
    'save': True,
    'save_period': 10,
    'device': 'cpu',   
    'workers': 2,
    'project': 'container_id_training',
    'name': 'yolov8_container_detectionV2',
    'exist_ok': True
}

print("\nTraining Configuration:")
for k, v in training_config.items():
    print(f" - {k}: {v}")

if os.path.exists(checkpoint_path):
    print(f"\nFound checkpoint: {checkpoint_path}")
    print("Resuming training from last saved state...\n")
    model = YOLO(checkpoint_path)
    try:
        results = model.train(resume=True)
        print("\nTraining resumed successfully!")
    except Exception as e:
        print(f"\nError while resuming training: {e}")
else:
    print(f"\nNo checkpoint found at {checkpoint_path}")
    print("Starting new training from scratch using base model.\n")
    model = YOLO(base_model)
    try:
        results = model.train(**training_config)
        print("\nNew training started successfully!")
    except Exception as e:
        print(f"\nTraining error: {e}")

print("\nTraining process finished.")
