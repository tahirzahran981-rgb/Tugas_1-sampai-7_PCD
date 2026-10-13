import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

FOLDER_INPUT = "images"   # folder tempat gambar ijazah
FOLDER_HASIL = "hasil"    # folder untuk menyimpan hasil
os.makedirs(FOLDER_HASIL, exist_ok=True)


# ---------------------------------------------------------------
# LANGKAH 1: Fungsi untuk mendeteksi masalah pada gambar
# ---------------------------------------------------------------
def deteksi_masalah(gray):
    """Mengembalikan daftar masalah dari gambar grayscale."""
    masalah = []

    rata_rata = gray.mean()   # kecerahan rata-rata (0 = hitam, 255 = putih)
    kontras = gray.std()      # sebaran intensitas (kecil = kontras rendah)

    if rata_rata < 90:
        masalah.append("terlalu gelap")
    if rata_rata > 200:
        masalah.append("terlalu terang")
    if kontras < 40:
        masalah.append("kontras rendah")

    # Cahaya tidak merata: bandingkan kecerahan 4 bagian gambar
    t, l = gray.shape
    bagian = [gray[:t//2, :l//2], gray[:t//2, l//2:],
              gray[t//2:, :l//2], gray[t//2:, l//2:]]
    kecerahan = [b.mean() for b in bagian]
    if max(kecerahan) - min(kecerahan) > 30:
        masalah.append("pencahayaan tidak merata")

    # Noise: selisih gambar dengan versi yang sudah dihaluskan
    noise = np.abs(gray.astype(float) - cv2.medianBlur(gray, 3)).mean()
    if noise > 3:
        masalah.append("noise")

    return masalah, rata_rata, kontras, noise


# ---------------------------------------------------------------
# LANGKAH 2: Fungsi untuk memperbaiki gambar sesuai masalahnya
# ---------------------------------------------------------------
def perbaiki(gray, masalah):
    """Mengembalikan gambar hasil perbaikan dan daftar metode yang dipakai."""
    hasil = gray.copy()
    metode = []

    # a. Noise -> filter median (dilakukan pertama agar noise tidak ikut diperkuat)
    if "noise" in masalah:
        hasil = cv2.medianBlur(hasil, 3)
        metode.append("Median Filter")

    # b. Cahaya tidak merata -> bagi gambar dengan latar belakang yang di-blur
    if "pencahayaan tidak merata" in masalah:
        latar = cv2.GaussianBlur(hasil, (101, 101), 0).astype(float)
        hasil = np.clip(hasil / (latar + 1) * 200, 0, 255).astype(np.uint8)
        metode.append("Koreksi Iluminasi")

    # c. Gelap / terang -> koreksi gamma
    if "terlalu gelap" in masalah:
        hasil = (255 * (hasil / 255.0) ** 0.5).astype(np.uint8)   # gamma < 1 = lebih terang
        metode.append("Gamma 0.5 (cerahkan)")
    if "terlalu terang" in masalah:
        hasil = (255 * (hasil / 255.0) ** 2.0).astype(np.uint8)   # gamma > 1 = lebih gelap
        metode.append("Gamma 2.0 (gelapkan)")

    # d. Kontras rendah -> CLAHE
    if "kontras rendah" in masalah:
        hasil = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(hasil)
        metode.append("CLAHE")

    return hasil, metode


# ---------------------------------------------------------------
# LANGKAH 3: Proses semua gambar
# ---------------------------------------------------------------
daftar_file = sorted(f for f in os.listdir(FOLDER_INPUT)
                     if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))

if not daftar_file:
    print(f"Tidak ada gambar di folder '{FOLDER_INPUT}'")

for nomor, nama_file in enumerate(daftar_file, start=1):
    # Baca gambar (OpenCV membaca dalam urutan BGR, kita ubah ke RGB)
    bgr = cv2.imread(os.path.join(FOLDER_INPUT, nama_file))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # RGB -> Grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Deteksi masalah dan perbaiki
    masalah, rata_rata, kontras, noise = deteksi_masalah(gray)
    hasil, metode = perbaiki(gray, masalah)

    # Cetak laporan singkat
    print(f"\n[{nomor}] {nama_file}")
    print(f"    Rata-rata kecerahan : {rata_rata:.1f}")
    print(f"    Kontras (std)       : {kontras:.1f}")
    print(f"    Masalah             : {', '.join(masalah) or 'tidak ada'}")
    print(f"    Metode perbaikan    : {', '.join(metode) or 'tidak perlu'}")

    # -----------------------------------------------------------
    # LANGKAH 4: Tampilkan gambar dan histogram
    # -----------------------------------------------------------
    plt.figure(figsize=(14, 7))
    plt.suptitle(f"{nama_file}  |  Masalah: {', '.join(masalah) or 'tidak ada'}")

    # Baris atas: gambar
    plt.subplot(2, 3, 1); plt.imshow(rgb);                   plt.title("Asli (RGB)");        plt.axis("off")
    plt.subplot(2, 3, 2); plt.imshow(gray, cmap="gray");     plt.title("Grayscale");         plt.axis("off")
    plt.subplot(2, 3, 3); plt.imshow(hasil, cmap="gray");    plt.title("Hasil perbaikan");   plt.axis("off")

    # Baris bawah: histogram
    plt.subplot(2, 3, 4)
    for i, warna in enumerate(["red", "green", "blue"]):
        plt.hist(rgb[:, :, i].ravel(), bins=256, range=(0, 255), color=warna, alpha=0.5, label=warna.upper())
    plt.title("Histogram RGB (asli)"); plt.legend()

    plt.subplot(2, 3, 5)
    plt.hist(gray.ravel(), bins=256, range=(0, 255), color="gray")
    plt.title("Histogram Grayscale")

    plt.subplot(2, 3, 6)
    plt.hist(hasil.ravel(), bins=256, range=(0, 255), color="green")
    plt.title("Histogram Hasil Perbaikan")

    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER_HASIL, f"hasil_{nomor}.png"), dpi=100)
    plt.close()

print(f"\nSelesai! Lihat hasilnya di folder '{FOLDER_HASIL}'.")
