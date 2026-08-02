
import pandas as pd
import matplotlib.pyplot as plt

file_path = "results.csv"

data = pd.read_csv(file_path)

plt.figure()
plt.plot(data['epoch'], data['train/box_loss'], label='Train Box Loss')
plt.plot(data['epoch'], data['val/box_loss'], label='Val Box Loss')
plt.plot(data['epoch'], data['train/cls_loss'], label='Train Cls Loss')
plt.plot(data['epoch'], data['val/cls_loss'], label='Val Cls Loss')
plt.plot(data['epoch'], data['train/dfl_loss'], label='Train DFL Loss')
plt.plot(data['epoch'], data['val/dfl_loss'], label='Val DFL Loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Grafik Loss YOLOv8")
plt.legend()
plt.show()

plt.figure()
plt.plot(data['epoch'], data['metrics/precision(B)'], label='Precision')
plt.plot(data['epoch'], data['metrics/recall(B)'], label='Recall')

plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Grafik Precision dan Recall")
plt.legend()
plt.show()

plt.figure()
plt.plot(data['epoch'], data['metrics/mAP50(B)'], label='mAP@0.5')
plt.plot(data['epoch'], data['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')

plt.xlabel("Epoch")
plt.ylabel("mAP")
plt.title("Grafik Mean Average Precision (mAP)")
plt.legend()
plt.show()
