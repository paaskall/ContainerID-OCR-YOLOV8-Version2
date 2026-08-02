import pandas as pd
import matplotlib.pyplot as plt

csv_path = "container_id_training/yolov8_container_detectionV2/results.csv"

df = pd.read_csv(csv_path)
print(df.head())

plt.figure(figsize=(12,8))
plt.subplot(2, 1, 1)
plt.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss", linewidth=2)
plt.plot(df["epoch"], df["train/cls_loss"], label="Train Class Loss", linewidth=2)
plt.title("YOLOv8 Training Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Grafik AKURASI
plt.subplot(2, 1, 2)
plt.plot(df["epoch"], df["metrics/precision(B)"], label="Precision", linewidth=2)
plt.plot(df["epoch"], df["metrics/recall(B)"], label="Recall", linewidth=2)
plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50", linewidth=2)
plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95", linewidth=2)

plt.title("YOLOv8 Evaluation Metrics")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.grid(True)

metrics = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
colors = ['C0', 'C1', 'C2', 'C3']  

for i, metric in enumerate(metrics):
    last_value = df[metric].iloc[-1]
    plt.text(df["epoch"].iloc[-1] + 0.5, last_value,
             f"{last_value*100:.1f}%",
             color=colors[i], fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()
