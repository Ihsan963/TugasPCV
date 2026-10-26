
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH_GAMBAR = "gambar.jpg"


def buat_gambar_contoh():
    img = np.full((256, 256), 60, dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (140, 140), 200, -1)
    cv2.circle(img, (180, 170), 50, 150, -1)
    cv2.putText(img, "TI", (60, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)
    return img


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


def konvolusi(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    H, W = img.shape
    padded = np.pad(img.astype(np.float64), ((ph, ph), (pw, pw)), mode="reflect")
    k = np.flipud(np.fliplr(kernel))
    out = np.zeros((H, W), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += k[i, j] * padded[i:i + H, j:j + W]
    return out


def kernel_mean(n):
    return np.ones((n, n)) / (n * n)


def kernel_gaussian(n, sigma):
    ax = np.arange(n) - n // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return k / k.sum()  


def filter_median(img, n=3):
    p = n // 2
    H, W = img.shape
    padded = np.pad(img, ((p, p), (p, p)), mode="reflect")
    tumpukan = [padded[i:i + H, j:j + W] for i in range(n) for j in range(n)]
    return np.median(np.stack(tumpukan, axis=0), axis=0).astype(np.uint8)


def tambah_salt_pepper(img, jumlah=0.05, seed=0):
    rng = np.random.default_rng(seed)
    hasil = img.copy()
    acak = rng.random(img.shape)
    hasil[acak < jumlah / 2] = 0
    hasil[acak > 1 - jumlah / 2] = 255
    return hasil


noisy = tambah_salt_pepper(gray, 0.05)

smooth = {
    "Asli + noise": noisy,
    "Mean 3x3": ke_uint8(konvolusi(noisy, kernel_mean(3))),
    "Mean 7x7": ke_uint8(konvolusi(noisy, kernel_mean(7))),
    "Gaussian 5x5 s=1": ke_uint8(konvolusi(noisy, kernel_gaussian(5, 1.0))),
    "Median 3x3": filter_median(noisy, 3),
}

plt.figure(figsize=(14, 3.5))
for i, (nama, im) in enumerate(smooth.items()):
    plt.subplot(1, 5, i + 1)
    tampil(im, nama)
plt.tight_layout()
plt.show()


lap_neg = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]], dtype=np.float64)
lap_8 = np.array([[1, 1, 1],
                  [1, -8, 1],
                  [1, 1, 1]], dtype=np.float64)

lap_hasil = konvolusi(gray, lap_neg)
sharp = ke_uint8(gray.astype(np.float64) - lap_hasil)

blur = konvolusi(gray, kernel_gaussian(5, 1.5))
unsharp = ke_uint8(gray + 1.5 * (gray - blur))

plt.figure(figsize=(12, 3.5))
for i, (nama, im) in enumerate([
    ("Asli", gray),
    ("Laplacian (diskalakan)", ke_uint8(np.abs(lap_hasil) * 3)),
    ("Sharpen (asli - Laplacian)", sharp),
    ("Unsharp masking", unsharp),
]):
    plt.subplot(1, 4, i + 1)
    tampil(im, nama)
plt.tight_layout()
plt.show()


sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)
sobel_y = sobel_x.T
prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float64)
prewitt_y = prewitt_x.T


def gradien(img, kx, ky):
    gx = konvolusi(img, kx)
    gy = konvolusi(img, ky)
    return np.sqrt(gx ** 2 + gy ** 2) 


halus = ke_uint8(konvolusi(gray, kernel_gaussian(5, 1.0))) 
tepi_sobel = gradien(halus, sobel_x, sobel_y)
tepi_prewitt = gradien(halus, prewitt_x, prewitt_y)

plt.figure(figsize=(12, 3.5))
for i, (nama, im) in enumerate([
    ("Asli", gray),
    ("Sobel", ke_uint8(tepi_sobel)),
    ("Prewitt", ke_uint8(tepi_prewitt)),
    ("Laplacian 8-tetangga", ke_uint8(np.abs(konvolusi(halus, lap_8)))),
]):
    plt.subplot(1, 4, i + 1)
    tampil(im, nama)
plt.tight_layout()
plt.show()


ref = cv2.blur(gray, (3, 3))
manual = ke_uint8(np.round(konvolusi(gray, kernel_mean(3))))
print("Selisih maksimum mean manual vs cv2.blur:", np.abs(ref.astype(int) - manual.astype(int)).max())
