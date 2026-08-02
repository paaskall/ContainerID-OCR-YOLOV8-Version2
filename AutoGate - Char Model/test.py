import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO

def xyxy_to_xywh(xyxy):
    x1, y1, x2, y2 = xyxy
    return x1, y1, x2 - x1, y2 - y1

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def nms_charwise(dets, iou_th=0.25):
    dets = sorted(dets, key=lambda d: d["conf"], reverse=True)
    keep = []
    for d in dets:
        ok = True
        for k in keep:
            if iou_xyxy(d["xyxy"], k["xyxy"]) > iou_th:
                ok = False
                break
        if ok:
            keep.append(d)
    return keep

def dedup_by_slots_ttb(dets, slot_gap_ratio=0.55):
    if not dets:
        return []

    # sort by center y
    dets = sorted(dets, key=lambda d: ((d["xyxy"][1] + d["xyxy"][3]) / 2.0))

    heights = [max(1.0, d["xyxy"][3] - d["xyxy"][1]) for d in dets]
    med_h = float(np.median(heights))
    gap = max(6.0, med_h * slot_gap_ratio)

    slots = []  # list of list[dets]
    cur = [dets[0]]
    cur_cy = (dets[0]["xyxy"][1] + dets[0]["xyxy"][3]) / 2.0

    for d in dets[1:]:
        cy = (d["xyxy"][1] + d["xyxy"][3]) / 2.0
        if abs(cy - cur_cy) <= gap:
            cur.append(d)
            cur_cy = (cur_cy + cy) / 2.0
        else:
            slots.append(cur)
            cur = [d]
            cur_cy = cy
    slots.append(cur)

    picked = [max(s, key=lambda x: x["conf"]) for s in slots]
    return picked

def build_text_from_dets(dets):
    return "".join([d["name"] for d in dets])

def draw_dets(img, dets, names, out_path):
    vis = img.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(vis, f"{d['name']} {d['conf']:.2f}", (x1, max(0,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/remote-user/AutoGate/AutoGate - Char Model/YoloV8_ImG_Proses/best.pt")
    ap.add_argument("--source", default="/home/remote-user/AutoGate/AutoGate - Char Model/test_images")
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou", type=float, default=0.30, help="NMS IoU (lebih kecil = lebih agresif)")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--reading_order", default="ttb", choices=["ttb", "ltr"])
    ap.add_argument("--save_img", action="store_true")
    ap.add_argument("--out", default="runs_char_test")
    args = ap.parse_args()

    model = YOLO(args.model)
    names = model.names

    exts = (".jpg",".jpeg",".png",".bmp",".webp")
    if os.path.isdir(args.source):
        imgs = [os.path.join(args.source, f) for f in sorted(os.listdir(args.source)) if f.lower().endswith(exts)]
    else:
        imgs = [args.source]

    print(f"Model: {args.model}")
    print(f"Total images: {len(imgs)}")
    print(f"Device: {args.device}")
    print(f"reading_order: {args.reading_order}")
    print(f"conf={args.conf} iou={args.iou} imgsz={args.imgsz}")
    print("-"*60)

    total_det = 0
    for idx, p in enumerate(imgs, 1):
        img = cv2.imread(p)
        if img is None:
            continue

        # YOLO predict (perketat NMS via iou)
        res = model.predict(
            source=img,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
            agnostic_nms=True,   # penting untuk char (bbox dobel lintas kelas sering terjadi)
            max_det=1000
        )[0]

        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                conf = float(b.conf[0].cpu().numpy())
                cls = int(b.cls[0].cpu().numpy())
                dets.append({
                    "xyxy": xyxy,
                    "conf": conf,
                    "cls": cls,
                    "name": names.get(cls, str(cls))
                })

        dets = nms_charwise(dets, iou_th=max(0.15, args.iou - 0.05))

        if args.reading_order == "ttb":
            dets = dedup_by_slots_ttb(dets, slot_gap_ratio=0.55)
            dets = sorted(dets, key=lambda d: ((d["xyxy"][1]+d["xyxy"][3])/2.0))
        else:
            dets = sorted(dets, key=lambda d: ((d["xyxy"][0]+d["xyxy"][2])/2.0))

        text = build_text_from_dets(dets)
        total_det += len(dets)

        print(f"[{idx}/{len(imgs)}] {os.path.basename(p)} | det={len(dets)} | text='{text}'")

        if args.save_img:
            out_path = os.path.join(args.out, "pred_images", os.path.basename(p))
            draw_dets(img, dets, names, out_path)

    print("-"*60)
    print(f"Done. total_detections={total_det}")
    print(f"Output folder: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
