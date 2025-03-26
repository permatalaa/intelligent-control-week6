from ultralytics import YOLO
import cv2
import numpy as np

# Load model YOLOv8 Instance Segmentation
model = YOLO("yolov8n-seg.pt")

def detect_rail_lane(image_path):
    """Mendeteksi jalur rel menggunakan YOLOv8 Instance Segmentation dan menghitung panjangnya"""
    image = cv2.imread(image_path)
    results = model(image)  # Jalankan deteksi tanpa show=True agar bisa diproses lebih lanjut
    
    if results[0].masks is not None:
        mask_coords = results[0].masks.xy  # Mendapatkan koordinat mask
    
        total_length = 0
        for coords in mask_coords:
            coords = np.array(coords, dtype=np.int32)  # Konversi ke numpy array
            length = cv2.arcLength(coords, closed=False)  # Hitung panjang kontur
            total_length += length

            # Gambar jalur deteksi pada gambar asli
            cv2.polylines(image, [coords], isClosed=False, color=(0, 255, 0), thickness=2)

        print(f"Panjang total jalur rel yang terdeteksi: {total_length:.2f} piksel")

    # Tampilkan gambar dengan hasil deteksi
    cv2.imshow("Deteksi Jalur Rel", image)
    cv2.waitKey(0)  # Tunggu tombol ditekan sebelum keluar
    cv2.destroyAllWindows()

# Contoh penggunaan
detect_rail_lane("dataset.jpg")
