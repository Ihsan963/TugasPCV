
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH_GAMBAR = "gambar.jpg"


def buat_gambar_contoh():
    y, x = np.mgrid[0:256, 0:256]
    g = (x * 0.35 + 60 + 20 * np.sin(y / 15.0)).astype(np.uint8)
    cv2.circle(g, (128, 128), 50, 140, -1)
    return g


def tampil(img, judul):
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.title(judul, fontsize=9)
    plt.axis("off")


def ke_uint8(arr):
    return np.clip(arr, 0, 255).astype(np.uint8)


img_bgr = cv2.imread(PATH_GAMBAR)
if img_bgr is None:
    print(f"'{PATH_GAMBAR}' tidak ditemukan -> memakai gambar contoh.")
    gray = buat_gambar_contoh()
else:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def negatif(img):
    return 255 - img


def log_transform(img, c=None):
    r = img.astype(np.float64)
    if c is None:
        c = 255 / np.log(1 + r.max())
    return ke_uint8(c * np.log(1 + r))


def gamma_transform(img, gamma, c=1.0):
    r = img.astype(np.float64) / 255.0
    return ke_uint8(c * 255 * (r ** gamma))


def contrast_stretching(img):
    r = img.astype(np.float64)
    rmin, rmax = r.min(), r.max()
    return ke_uint8((r - rmin) / (rmax - rmin) * 255)


def thresholding(img, T=127):
    return np.where(img > T, 255, 0).astype(np.uint8)


def piecewise_linear(img, r1, s1, r2, s2):
    r = img.astype(np.float64)
    s = np.zeros_like(r)
    a = r <= r1
    b = (r > r1) & (r <= r2)
    c = r > r2
    s[a] = (s1 / r1) * r[a]
    s[b] = ((s2 - s1) / (r2 - r1)) * (r[b] - r1) + s1
    s[c] = ((255 - s2) / (255 - r2)) * (r[c] - r2) + s2
    return ke_uint8(s)


hasil = {
    "Asli": gray,
    "Negatif": negatif(gray),
    "Log": log_transform(gray),
    "Gamma 0.5 (cerah)": gamma_transform(gray, 0.5),
    "Gamma 2.0 (gelap)": gamma_transform(gray, 2.0),
    "Contrast stretching": contrast_stretching(gray),
    "Threshold T=127": thresholding(gray, 127),
    "Piecewise linear": piecewise_linear(gray, 70, 30, 180, 220),
}

plt.figure(figsize=(14, 6))
for i, (nama, im) in enumerate(hasil.items()):
    plt.subplot(2, 4, i + 1)
    tampil(im, nama)
plt.tight_layout()
plt.show()


def hitung_histogram(img):
    """Hitung jumlah piksel untuk tiap level 0..255 dengan loop manual."""
    hist = np.zeros(256, dtype=np.int64)
    for nilai in img.ravel():
        hist[nilai] += 1
    return hist


def ekualisasi_histogram(img):
    hist = hitung_histogram(img)
    pdf = hist / img.size
    cdf = np.cumsum(pdf)
    tabel = np.round(255 * cdf).astype(np.uint8)
    return tabel[img], hist, pdf, cdf, tabel


eq, hist_asli, pdf, cdf, tabel = ekualisasi_histogram(gray)
hist_eq = hitung_histogram(eq)

plt.figure(figsize=(12, 7))
plt.subplot(2, 3, 1)
tampil(gray, "Citra asli")
plt.subplot(2, 3, 2)
tampil(eq, "Hasil ekualisasi")
plt.subplot(2, 3, 3)
plt.plot(cdf, label="CDF")
plt.plot(tabel / 255.0, "--", label="Fungsi transformasi s (dinormalisasi)")
plt.title("CDF & fungsi transformasi", fontsize=9)
plt.legend(fontsize=7)
plt.subplot(2, 3, 4)
plt.bar(range(256), hist_asli, width=1)
plt.title("Histogram asli", fontsize=9)
plt.subplot(2, 3, 5)
plt.bar(range(256), hist_eq, width=1, color="tab:orange")
plt.title("Histogram setelah ekualisasi", fontsize=9)
plt.tight_layout()
plt.show()


if img_bgr is not None:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = ekualisasi_histogram(hsv[:, :, 2])[0]
    berwarna_eq = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title("Asli")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(berwarna_eq)
    plt.title("Ekualisasi kanal V")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
