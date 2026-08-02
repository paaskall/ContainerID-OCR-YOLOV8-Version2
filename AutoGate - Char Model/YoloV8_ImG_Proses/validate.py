from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os

def show_predictions(model, image_folder, num_images=5):
    
    images = os.listdir(image_folder)[:num_images]

    plt.figure(figsize=(15,6))

    for i, img_name in enumerate(images):

        path = os.path.join(image_folder, img_name)

        results = model.predict(path, conf=0.25)

        img = results[0].plot()

        plt.subplot(1, num_images, i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")

    plt.suptitle("Sample Predictions")
    plt.show()


def main():

    model = YOLO("best.pt")

    print("\n===== VALIDATION =====")

    metrics = model.val(
        data="data.yaml",
        split="val",
        imgsz=640,
        plots=True
    )

    print("\n===== METRICS =====")

    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)

    print("\n===== TEST DATASET =====")

    test_metrics = model.val(
        data="data.yaml",
        split="test",
        imgsz=640,
        plots=True
    )

    print("Test mAP50:", test_metrics.box.map50)

    print("\n===== SHOW SAMPLE PREDICTIONS =====")

    show_predictions(model, "test/images", 5)


if __name__ == "__main__":
    main()