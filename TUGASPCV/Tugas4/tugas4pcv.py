
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH_GAMBAR = "gambar.jpg"
EPS = 1e-8  


def buat_gambar_contoh():
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    for x in range(w):  
        hue = int(179 * x / w)
        img[:, x] = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    img[150:, :] = np.linspace(0, 255, w, dtype=np.uint8)[None, :, None] 
    return img


def tampil(img, judul, gray=True):
    if gray:
        plt.imshow(img, cmap="gray", vmin=0, vmax=1)
    else:
        plt.imshow(img)
    plt.title(judul, fontsize=9)
    plt.axis("off")

img_bgr = cv2.imread(PATH_GAMBAR)
if img_bgr is None:
    print(f"'{PATH_GAMBAR}' tidak ditemukan -> memakai gambar contoh.")
    img_bgr = buat_gambar_contoh()

rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
R, G, B = [rgb[:, :, i].astype(np.float64) / 255.0 for i in range(3)]


def rgb_ke_cmyk(R, G, B):
    K = 1 - np.maximum(np.maximum(R, G), B)
    penyebut = 1 - K
    aman = penyebut > EPS  
    C = np.where(aman, (1 - R - K) / np.where(aman, penyebut, 1), 0)
    M = np.where(aman, (1 - G - K) / np.where(aman, penyebut, 1), 0)
    Y = np.where(aman, (1 - B - K) / np.where(aman, penyebut, 1), 0)
    return C, M, Y, K


C, M, Y, K = rgb_ke_cmyk(R, G, B)


def rgb_ke_hsv(R, G, B):
    mx = np.maximum(np.maximum(R, G), B)
    mn = np.minimum(np.minimum(R, G), B)
    delta = mx - mn

    V = mx
    S = np.where(mx > EPS, delta / np.where(mx > EPS, mx, 1), 0)

    d = np.where(delta > EPS, delta, 1)
    H = np.zeros_like(mx)
    H = np.where(mx == R, 60 * (((G - B) / d) % 6), H)
    H = np.where((mx == G) & (mx != R), 60 * (((B - R) / d) + 2), H)
    H = np.where((mx == B) & (mx != R) & (mx != G), 60 * (((R - G) / d) + 4), H)
    H = np.where(delta > EPS, H, 0) 
    return H, S, V


H_hsv, S_hsv, V_hsv = rgb_ke_hsv(R, G, B)



def rgb_ke_hsi(R, G, B):
    total = R + G + B
    I = total / 3.0
    S = np.where(total > EPS, 1 - 3 * np.minimum(np.minimum(R, G), B) / np.where(total > EPS, total, 1), 0)

    atas = 0.5 * ((R - G) + (R - B))
    bawah = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
    cos_t = np.where(bawah > EPS, atas / np.where(bawah > EPS, bawah, 1), 1)
    theta = np.degrees(np.arccos(np.clip(cos_t, -1, 1)))
    H = np.where(B <= G, theta, 360 - theta)
    H = np.where(bawah > EPS, H, 0)
    return H, S, I


H_hsi, S_hsi, I_hsi = rgb_ke_hsi(R, G, B)


plt.figure(figsize=(14, 9))
plt.subplot(4, 4, 1)
tampil(rgb, "RGB asli", gray=False)
for i, (kanal, nama) in enumerate([(R, "R"), (G, "G"), (B, "B")]):
    plt.subplot(4, 4, 2 + i)
    tampil(kanal, f"RGB - {nama}")

for i, (kanal, nama) in enumerate([(C, "C"), (M, "M"), (Y, "Y"), (K, "K")]):
    plt.subplot(4, 4, 5 + i)
    tampil(kanal, f"CMYK - {nama}")

for i, (kanal, nama, mx) in enumerate([(H_hsv, "H", 360), (S_hsv, "S", 1), (V_hsv, "V", 1)]):
    plt.subplot(4, 4, 9 + i)
    tampil(kanal / mx, f"HSV - {nama}")

for i, (kanal, nama, mx) in enumerate([(H_hsi, "H", 360), (S_hsi, "S", 1), (I_hsi, "I", 1)]):
    plt.subplot(4, 4, 13 + i)
    tampil(kanal / mx, f"HSI - {nama}")
plt.tight_layout()
plt.show()


hsv_cv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
err_h = np.abs(hsv_cv[:, :, 0] - H_hsv / 2).mean()
err_s = np.abs(hsv_cv[:, :, 1] - S_hsv * 255).mean()
err_v = np.abs(hsv_cv[:, :, 2] - V_hsv * 255).mean()
print(f"Rata-rata selisih vs OpenCV -> H: {err_h:.3f}, S: {err_s:.3f}, V: {err_v:.3f} (kecil = benar)")


R2, G2, B2 = (1 - C) * (1 - K), (1 - M) * (1 - K), (1 - Y) * (1 - K)
print("Error CMYK -> RGB:", np.abs(np.dstack([R2, G2, B2]) - np.dstack([R, G, B])).max())


def hsv_ke_rgb(H, S, V):
    Cc = V * S
    X = Cc * (1 - np.abs((H / 60.0) % 2 - 1))
    m = V - Cc
    sektor = (H // 60).astype(int) % 6
    r = np.choose(sektor, [Cc, X, 0 * Cc, 0 * Cc, X, Cc])
    g = np.choose(sektor, [X, Cc, Cc, X, 0 * Cc, 0 * Cc])
    b = np.choose(sektor, [0 * Cc, 0 * Cc, X, Cc, Cc, X])
    return r + m, g + m, b + m


R3, G3, B3 = hsv_ke_rgb(H_hsv, S_hsv, V_hsv)
print("Error HSV -> RGB :", np.abs(np.dstack([R3, G3, B3]) - np.dstack([R, G, B])).max())
