from ultralytics import YOLO

model = YOLO("best.pt")

metrics = model.val(
    data="/home/remote-user/AutoGate/models/data.yaml"
)

# Cetak hasil per class
for i, c in enumerate(model.names.values()):
    print(f"\nClass: {c}")
    print(f"Precision: {metrics.box.p[i]:.3f}")
    print(f"Recall: {metrics.box.r[i]:.3f}")
    print(f"mAP50: {metrics.box.ap50[i]:.3f}")
    print(f"mAP50-95: {metrics.box.ap[i]:.3f}")
