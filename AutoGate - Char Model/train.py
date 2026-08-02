import os
from ultralytics import YOLO

def main():
    device = 0

    data_yaml = os.path.join("Kall-s Dataset.v2i.yolov8", "data.yaml")

    model = YOLO("yolov8s.pt")

    model.train(
        data=data_yaml,
        imgsz=640,
        epochs=100,
        batch=16,
        workers=4,
        device=device,
        project="runs_char",
        name="yolov8s_char_det",
        patience=20,
        cache=True,
        verbose=True
    )

    model.val(data=data_yaml, device=device)

if __name__ == "__main__":
    main()
