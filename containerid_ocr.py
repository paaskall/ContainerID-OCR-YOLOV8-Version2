from ultralytics import YOLO
import cv2
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import google.generativeai as genai
from PIL import Image
import time
import re

# Configuration
MODEL_PATH = "container_id_training/yolov8_container_detectionOBB/weights/best.pt"
IMAGE_DIR = "images"
OUTPUT_CSV = "result/hasil_ocr.csv"
GEMINI_API_KEY = "AIzaSyDgr5oSJad23oCsIL4xKcarshDzicdqlfk"

model = YOLO(MODEL_PATH)

genai.configure(api_key=GEMINI_API_KEY)

GEMINI_MODEL_NAME = "models/gemini-2.5-flash-preview-05-20"

try:
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    print(f"Model Gemini berhasil diinisialisasi: {GEMINI_MODEL_NAME}")
    
    # Test koneksi
    test_response = gemini_model.generate_content("Test")
    print("Test koneksi berhasil")
    
except Exception as e:
    print(f"✗ Error inisialisasi Gemini: {e}")
    print("Pastikan API key valid dan model tersedia")
    exit(1)

print(f"YOLOv8 + Gemini OCR ({GEMINI_MODEL_NAME}) dengan Complementary Reading initialized.\n")

def preprocess_for_gemini(roi):
    """Preprocessing untuk Gemini"""
    if len(roi.shape) == 3 and roi.shape[2] == 3:
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    else:
        roi_rgb = roi
    
    # Resize
    height, width = roi_rgb.shape[:2]
    if height < 50 or width < 50:
        scale = max(150/height, 150/width)
        new_height = int(height * scale)
        new_width = int(width * scale)
        roi_rgb = cv2.resize(roi_rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    return roi_rgb

def image_to_pil(roi_array):
    """Convert numpy array ke PIL Image"""
    roi_processed = preprocess_for_gemini(roi_array)
    return Image.fromarray(roi_processed)

def ocr_with_gemini_only(roi_image, ocr_type="container_id"):
    """OCR menggunakan Google Gemini"""
    try:
        if ocr_type == "container_id":
            prompt = """Read the Container ID from this image. 
            Format should be 4 letters followed by 7 numbers (like MSCU1234567).
            Return ONLY the Container ID text, or 'TIDAK_TERBACA' if not readable."""
        
        elif ocr_type == "license_plate":
            prompt = """Read the license plate number from this image. 
            Return ONLY the license plate text, or 'TIDAK_TERBACA' if not readable."""
        
        else:
            prompt = """Read the text from this image. 
            Return ONLY the text content, or 'TIDAK_TERBACA' if not readable."""

        pil_image = image_to_pil(roi_image)
        
        response = gemini_model.generate_content([prompt, pil_image])
        
        if response and response.text:
            ocr_text = response.text.strip()
            
            if ocr_type == "container_id":
                clean_text = re.sub(r'[^A-Z0-9]', '', ocr_text.upper())
                container_pattern = r'[A-Z]{4}\d{7}'
                match = re.search(container_pattern, clean_text)
                
                if match:
                    return match.group(), "success"
                else:
                    if len(clean_text) >= 8 and "TIDAK" not in ocr_text.upper():
                        return clean_text, "success"
                    else:
                        return "TIDAK_TERBACA", "failed"
            
            # Validasi
            elif ocr_text.upper() != "TIDAK_TERBACA" and len(ocr_text) > 1:
                return ocr_text, "success"
            else:
                return "TIDAK_TERBACA", "failed"
        else:
            return "TIDAK_TERBACA", "error"
        
    except Exception as e:
        print(f"    Gemini OCR error: {str(e)}")
        return "TIDAK_TERBACA", "error"

def apply_complementary_reading(ocr_results, class_type):
    """
    Menerapkan complementary reading untuk results dalam satu gambar
    """
    print(f"    Applying complementary reading untuk {class_type}...")
    
    # Cari semua hasil yang success
    success_results = [result for result in ocr_results if result['ocr_status'] == 'success']
    
    if not success_results:
        print(f"      Tidak ada success results untuk {class_type}, skip complementary")
        for result in ocr_results:
            result['complementary_source'] = False
        return ocr_results
    
    replacement_text = success_results[0]['ocr_text']
    print(f"      Replacement text dari success: {replacement_text}")
    
    updated_results = []
    complementary_count = 0
    
    for result in ocr_results:
        if result['ocr_status'] != 'success':
            updated_result = result.copy()
            updated_result['ocr_text'] = replacement_text
            updated_result['complementary_source'] = True
            updated_result['original_status'] = result['ocr_status']
            updated_results.append(updated_result)
            complementary_count += 1
            print(f"      Detection {result['detection_id']} -> COMPLEMENTARY: {replacement_text}")
        else:
            updated_result = result.copy()
            updated_result['complementary_source'] = False
            updated_results.append(updated_result)
            print(f"      Detection {result['detection_id']} -> SUCCESS: {result['ocr_text']}")
    
    print(f"      Complementary applied: {complementary_count} detections diperbaiki")
    return updated_results

# PROSES SEMUA GAMBAR
results_data = []

for filename in os.listdir(IMAGE_DIR):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Gagal memuat gambar: {filename}")
        continue
        
    print(f"\n{'='*50}")
    print(f"Processing: {filename}")
    print(f"{'='*50}")

    detections = model(img)[0]
    print(f"  Ditemukan {len(detections.boxes)} objek")

    container_results = []
    license_plate_results = []
    other_results = []

    for i, box in enumerate(detections.boxes):
        cls_id = int(box.cls)
        label = model.names[cls_id]
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        pad = 10
        roi = img[max(0, y1-pad):min(img.shape[0], y2+pad),
                  max(0, x1-pad):min(img.shape[1], x2+pad)]

        if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
            print(f"   [{i+1}] ROI terlalu kecil, dilewati")
            continue

        if 'container' in label.lower():
            ocr_type = "container_id"
        elif 'plate' in label.lower() or 'plat' in label.lower():
            ocr_type = "license_plate"
        else:
            ocr_type = "general"

        print(f"   [{i+1}] Class: {label} | Conf: {conf:.2f} | OCR Type: {ocr_type}")

        ocr_text, ocr_status = ocr_with_gemini_only(roi, ocr_type)
        print(f"      Gemini OCR Result: {ocr_text} | Status: {ocr_status}")

        result_item = {
            "file": filename,
            "class": label,
            "confidence": conf,
            "ocr_text": ocr_text,
            "ocr_engine": "gemini", 
            "ocr_status": ocr_status,
            "bbox": (x1, y1, x2, y2),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "detection_id": i+1,
            "complementary_source": False,
            "original_status": ocr_status
        }

        if 'container' in label.lower():
            container_results.append(result_item)
        elif 'plate' in label.lower() or 'plat' in label.lower():
            license_plate_results.append(result_item)
        else:
            other_results.append(result_item)

    print(f"\n  COMPLEMENTARY READING:")
    print(f"    Sebelum: Container={len(container_results)}, License Plate={len(license_plate_results)}")
    
    if container_results:
        container_results = apply_complementary_reading(container_results, "container")
    
    if license_plate_results:
        license_plate_results = apply_complementary_reading(license_plate_results, "license_plate")
    
    print(f"    Setelah: Container={len(container_results)}, License Plate={len(license_plate_results)}")

    final_results = container_results + license_plate_results + other_results
    
    for result in final_results:
        x1, y1, x2, y2 = result['bbox']
        ocr_text = result['ocr_text']
        ocr_status = result['ocr_status']
        complementary_source = result['complementary_source']
        
        if ocr_status == "success":
            color = (0, 255, 0)  # Hijau - Success asli
            status_label = "SUCCESS"
        elif complementary_source:
            color = (255, 255, 0)  # Kuning - Complementary
            status_label = "COMPLEMENTARY"
        else:
            color = (0, 0, 255)  # Merah - Failed
            status_label = "FAILED"
        
        # Gambar bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Label text
        label_text = f"{result['class']}: {ocr_text} ({status_label})"
        cv2.putText(img, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        results_data.append(result)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"YOLO & OCR with Complementary - {filename}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if results_data:
    df = pd.DataFrame(results_data)
    df.to_csv(OUTPUT_CSV, index=False)
    
    total_count = len(df)
    success_count = len(df[df['ocr_status'] == 'success'])
    complementary_count = len(df[df['complementary_source'] == True])
    failed_count = len(df[(df['ocr_status'] == 'failed') & (df['complementary_source'] == False)])
    
    print(f"\n{'='*50}")
    print("FINAL STATISTICS:")
    print(f"{'='*50}")
    print(f"Total deteksi: {total_count}")
    print(f"Success asli: {success_count} ({success_count/total_count*100:.1f}%)")
    print(f"Complementary: {complementary_count} ({complementary_count/total_count*100:.1f}%)")
    print(f"Failed: {failed_count} ({failed_count/total_count*100:.1f}%)")
    print(f"Effective success rate: {(success_count + complementary_count)/total_count*100:.1f}%")
    print(f"\nHasil disimpan ke: {OUTPUT_CSV}")
    
    print(f"\nDETAIL HASIL:")
    for _, row in df.iterrows():
        status_icon = "✓" if row['ocr_status'] == 'success' else "🔄" if row['complementary_source'] else "✗"
        status_desc = "SUCCESS" if row['ocr_status'] == 'success' else "COMPLEMENTARY" if row['complementary_source'] else "FAILED"
        print(f"  {status_icon} {row['file']} - {row['class']}: '{row['ocr_text']}' ({status_desc})")
        
else:
    print("\nTidak ada data yang diproses.")