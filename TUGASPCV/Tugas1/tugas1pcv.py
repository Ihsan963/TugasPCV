
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH_GAMBAR = "gambar.jpg"
PATH_VIDEO = 0  


def buat_gambar_contoh():
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    for x in range(400):
        img[:, x] = (x * 255 // 399,) * 3
    cv2.rectangle(img, (30, 30), (130, 130), (0, 0, 255), -1)     
    cv2.rectangle(img, (150, 30), (250, 130), (0, 255, 0), -1)    
    cv2.rectangle(img, (270, 30), (370, 130), (255, 0, 0), -1)     
    cv2.rectangle(img, (30, 160), (130, 260), (0, 255, 255), -1)   
    cv2.circle(img, (260, 210), 50, (0, 128, 255), -1)             
    return img


def tampil(img_bgr, judul="", gray=False):
    if gray or img_bgr.ndim == 2:
        plt.imshow(img_bgr, cmap="gray", vmin=0, vmax=255)
    else:
        plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title(judul)
    plt.axis("off")

img = cv2.imread(PATH_GAMBAR)
if img is None:
    print(f"'{PATH_GAMBAR}' tidak ditemukan -> memakai gambar contoh.")
    img = buat_gambar_contoh()

print("Shape  :", img.shape, "(tinggi, lebar, kanal)")
print("Dtype  :", img.dtype)
print("Min/Max:", img.min(), img.max())
print("Piksel (0,0) [B G R]:", img[0, 0])


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
tampil(img, "Berwarna (RGB)")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(1, 2, 2)
tampil(gray, "Grayscale", gray=True)
plt.tight_layout()
plt.show()

b, g, r = cv2.split(img)
plt.figure(figsize=(12, 3))
for i, (kanal, nama) in enumerate([(r, "Red"), (g, "Green"), (b, "Blue")]):
    plt.subplot(1, 3, i + 1)
    tampil(kanal, f"Kanal {nama}", gray=True)
plt.tight_layout()
plt.show()


def filter_warna(img_bgr, lower, upper):
    hsv = cv2.cvtColor(cv2.GaussianBlur(img_bgr, (7, 7), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   
    hasil = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    return mask, hasil


rentang = {
    "Merah": ([0, 120, 70], [10, 255, 255]),
    "Hijau": ([40, 80, 70], [80, 255, 255]),
    "Biru": ([100, 120, 70], [130, 255, 255]),
    "Kuning": ([20, 100, 100], [35, 255, 255]),
}

plt.figure(figsize=(12, 6))
for i, (nama, (lo, hi)) in enumerate(rentang.items()):
    mask, hasil = filter_warna(img, lo, hi)
    plt.subplot(2, 4, i + 1)
    tampil(mask, f"Mask {nama}", gray=True)
    plt.subplot(2, 4, i + 5)
    tampil(hasil, f"Hasil {nama}")
plt.tight_layout()
plt.show()

LOWER = [90, 40, 50]
UPPER = [130, 255, 255]

cap = cv2.VideoCapture(PATH_VIDEO)
if not cap.isOpened():
    print("Video / webcam tidak bisa dibuka. Cek PATH_VIDEO.")
else:
    while True:
        ok, frame = cap.read()
        if not ok: 
            break
        mask, hasil = filter_warna(frame, LOWER, UPPER)
        cv2.imshow("Frame asli", frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("Hasil filter warna", hasil)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
