from ultralytics import YOLO

def main():

    # load model (pretrained)
    model = YOLO("yolov8n.pt")

    # train model
    model.train(
        data="data.yaml", 
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        project="yolo_training",
        name="yolov8_model"
    )

if __name__ == "__main__":
    main()