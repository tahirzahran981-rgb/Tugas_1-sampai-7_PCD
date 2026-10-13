import csv
from difflib import SequenceMatcher

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

# Jika di Windows dan Tesseract tidak terdeteksi, hapus tanda '#' di bawah
# lalu sesuaikan lokasinya:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ===============================================================
# PENGATURAN (ubah bagian ini saja)
# ===============================================================
FILE_GAMBAR = "nomor_ijazah.png"        # gambar nomor ijazah (hasil crop)
TEKS_BENAR = "No. 12345/IJZ/2020"       # nomor ijazah yang SEBENARNYA (kunci jawaban)
TAMBAH_NOISE = False                    # True = tambahkan noise salt-and-pepper buatan
PERBESAR = 3                            # perbesar citra 3x agar OCR lebih mudah


# ===============================================================
# LANGKAH 1: Baca citra & ubah ke grayscale
# ===============================================================
gambar = cv2.imread(FILE_GAMBAR)
if gambar is None:
    raise SystemExit(f"Gambar '{FILE_GAMBAR}' tidak ditemukan!")

gray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, None, fx=PERBESAR, fy=PERBESAR, interpolation=cv2.INTER_CUBIC)

# (Opsional) tambahkan noise salt-and-pepper: sebagian piksel jadi hitam/putih acak
if TAMBAH_NOISE:
    acak = np.random.default_rng(0).random(gray.shape)
    gray[acak < 0.03] = 0       # 3% piksel jadi hitam  (pepper)
    gray[acak > 0.97] = 255     # 3% piksel jadi putih  (salt)


# ===============================================================
# LANGKAH 2: Terapkan filter dan sharpening
# ===============================================================
# 1. Mean filter     : tiap piksel diganti rata-rata tetangganya (5x5)
mean = cv2.blur(gray, (5, 5))

# 2. Median filter   : tiap piksel diganti nilai TENGAH dari tetangganya (5x5)
median = cv2.medianBlur(gray, 5)

# 3. Gaussian filter : rata-rata berbobot (tetangga dekat bobotnya lebih besar)
gaussian = cv2.GaussianBlur(gray, (5, 5), 0)

# 4. Sharpening      : mempertajam tepi karakter memakai kernel
kernel_tajam = np.array([[ 0, -1,  0],
                         [-1,  5, -1],
                         [ 0, -1,  0]])
sharpening = cv2.filter2D(gray, -1, kernel_tajam)

# Tambahan: kombinasi median (hilangkan noise) lalu sharpening (pertajam tepi)
median_sharp = cv2.filter2D(median, -1, kernel_tajam)

hasil = {
    "Original": gray,
    "Mean filter": mean,
    "Median filter": median,
    "Gaussian filter": gaussian,
    "Sharpening": sharpening,
    "Median + Sharpening": median_sharp,
}


# ===============================================================
# LANGKAH 3: OCR + hitung akurasi
# ===============================================================
def hitung_karakter_benar(teks_ocr, teks_benar):
    """Menghitung jumlah karakter yang cocok antara hasil OCR dan teks sebenarnya."""
    cocok = SequenceMatcher(None, teks_ocr, teks_benar).get_matching_blocks()
    return sum(blok.size for blok in cocok)


tabel = []
for nama, img in hasil.items():
    # --psm 7 = anggap gambar berisi satu baris teks
    teks = pytesseract.image_to_string(img, config="--psm 7").strip()
    benar = hitung_karakter_benar(teks, TEKS_BENAR)
    akurasi = benar / len(TEKS_BENAR) * 100
    tabel.append((nama, teks, benar, akurasi))

# Cetak tabel
print(f"\nTeks sebenarnya: {TEKS_BENAR}  ({len(TEKS_BENAR)} karakter)\n")
print(f"{'Metode':<22}{'Hasil OCR':<26}{'Karakter Benar':>15}{'Akurasi':>10}")
print("-" * 73)
for nama, teks, benar, akurasi in tabel:
    print(f"{nama:<22}{teks:<26}{benar:>9}/{len(TEKS_BENAR):<5}{akurasi:>8.1f}%")

terbaik = max(tabel, key=lambda baris: baris[3])
print(f"\nMetode dengan akurasi tertinggi: {terbaik[0]} ({terbaik[3]:.1f}%)")

# Simpan tabel ke file CSV (bisa dibuka di Excel)
with open("tabel_tugas5.csv", "w", newline="", encoding="utf-8") as f:
    penulis = csv.writer(f)
    penulis.writerow(["Metode", "Hasil OCR", "Karakter Benar", "Akurasi (%)"])
    for nama, teks, benar, akurasi in tabel:
        penulis.writerow([nama, teks, f"{benar}/{len(TEKS_BENAR)}", f"{akurasi:.1f}"])


# ===============================================================
# LANGKAH 4: Tampilkan semua citra
# ===============================================================
plt.figure(figsize=(14, 7))
for i, (nama, teks, benar, akurasi) in enumerate(tabel):
    plt.subplot(3, 2, i + 1)
    plt.imshow(hasil[nama], cmap="gray", vmin=0, vmax=255)
    plt.title(f"{nama} | OCR: '{teks}' | {akurasi:.0f}%", fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.savefig("hasil_tugas5.png", dpi=110)
print("\nTersimpan: hasil_tugas5.png dan tabel_tugas5.csv")
plt.show()
