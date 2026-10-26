
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH_GAMBAR = "gambar.jpg"


def buat_gambar_contoh():
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (110, 110), 255, -1)
    cv2.circle(img, (150, 140), 35, 255, -1)
    rng = np.random.default_rng(1)
    for _ in range(60):   
        x, y = rng.integers(0, 200, 2)
        img[y, x] = 255
    for _ in range(40):   
        x, y = rng.integers(35, 105, 2)
        img[y, x] = 0
    return img


def tampil(img, judul):
    plt.imshow(img, cmap="gray", vmin=0, vmax=1)
    plt.title(judul, fontsize=9)
    plt.axis("off")



img_bgr = cv2.imread(PATH_GAMBAR)
if img_bgr is None:
    print(f"'{PATH_GAMBAR}' tidak ditemukan -> memakai gambar contoh.")
    gray = buat_gambar_contoh()
else:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

biner = (gray > 127).astype(np.uint8)


def se_persegi(n=3):
    return np.ones((n, n), dtype=np.uint8)


def se_silang(n=3):
    se = np.zeros((n, n), dtype=np.uint8)
    se[n // 2, :] = 1
    se[:, n // 2] = 1
    return se


def erosi(img, se):
    """Erosi: AND dari semua piksel yang tertutup oleh SE."""
    kh, kw = se.shape
    ph, pw = kh // 2, kw // 2
    H, W = img.shape
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="constant", constant_values=1)
    out = np.ones((H, W), dtype=np.uint8)
    for i in range(kh):
        for j in range(kw):
            if se[i, j]:
                out &= padded[i:i + H, j:j + W]
    return out


def dilasi(img, se):
    kh, kw = se.shape
    ph, pw = kh // 2, kw // 2
    H, W = img.shape
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="constant", constant_values=0)
    out = np.zeros((H, W), dtype=np.uint8)
    for i in range(kh):
        for j in range(kw):
            if se[i, j]:
                out |= padded[i:i + H, j:j + W]
    return out


def opening(img, se):
    return dilasi(erosi(img, se), se)


def closing(img, se):
    return erosi(dilasi(img, se), se)

se = se_persegi(3)

hasil = {
    "Biner asli": biner,
    "Erosi": erosi(biner, se),
    "Dilasi": dilasi(biner, se),
    "Opening (hapus noise putih)": opening(biner, se),
    "Closing (tutup lubang hitam)": closing(biner, se),
}

plt.figure(figsize=(14, 3.5))
for i, (nama, im) in enumerate(hasil.items()):
    plt.subplot(1, 5, i + 1)
    tampil(im, nama)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 3.5))
variasi = {
    "Erosi persegi 3x3": erosi(biner, se_persegi(3)),
    "Erosi persegi 7x7": erosi(biner, se_persegi(7)),
    "Erosi silang 3x3": erosi(biner, se_silang(3)),
    "Dilasi persegi 7x7": dilasi(biner, se_persegi(7)),
    "Dilasi silang 3x3": dilasi(biner, se_silang(3)),
}
for i, (nama, im) in enumerate(variasi.items()):
    plt.subplot(1, 5, i + 1)
    tampil(im, nama)
plt.tight_layout()
plt.show()

batas = dilasi(biner, se) - erosi(biner, se)
plt.figure(figsize=(4, 4))
tampil(batas, "Batas objek (dilasi - erosi)")
plt.show()


k = np.ones((3, 3), np.uint8)
for nama, manual, ref in [
    ("erosi", hasil["Erosi"], cv2.erode(biner, k)),
    ("dilasi", hasil["Dilasi"], cv2.dilate(biner, k)),
    ("opening", hasil["Opening (hapus noise putih)"], cv2.morphologyEx(biner, cv2.MORPH_OPEN, k)),
    ("closing", hasil["Closing (tutup lubang hitam)"], cv2.morphologyEx(biner, cv2.MORPH_CLOSE, k)),
]:
    print(f"{nama:8s} sama dengan OpenCV:", np.array_equal(manual, ref))
