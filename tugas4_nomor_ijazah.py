import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================================================
# PENGATURAN (ubah bagian ini saja)
# ===============================================================
FILE_GAMBAR = "ijazah.jpg"   # nama file gambar ijazah

# Koordinat area nomor ijazah: (x, y, lebar, tinggi) dalam piksel.
# Jika diisi None, akan muncul jendela untuk memilih area dengan mouse
# (drag area nomor ijazah, lalu tekan ENTER).
AREA_NOMOR = None            # contoh: (120, 80, 300, 40)

PERBESAR = 3                 # perbesar hasil crop 3x agar karakter lebih jelas


# ===============================================================
# LANGKAH 1: Baca gambar & crop area nomor ijazah
# ===============================================================
gambar = cv2.imread(FILE_GAMBAR)
if gambar is None:
    raise SystemExit(f"Gambar '{FILE_GAMBAR}' tidak ditemukan!")

if AREA_NOMOR is None:
    x, y, w, h = cv2.selectROI("Pilih area nomor ijazah lalu tekan ENTER", gambar)
    cv2.destroyAllWindows()
else:
    x, y, w, h = AREA_NOMOR

crop = gambar[y:y + h, x:x + w]                       # potong area
crop = cv2.resize(crop, None, fx=PERBESAR, fy=PERBESAR, interpolation=cv2.INTER_CUBIC)
asli = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)         # ubah ke grayscale


# ===============================================================
# LANGKAH 2: Tiga metode enhancement
# ===============================================================

# a) Brightness adjustment
#    Menambah/mengurangi nilai semua piksel dengan angka yang sama.
#    Di sini kecerahan digeser supaya rata-ratanya menjadi 127 (tengah).
geser = int(127 - asli.mean())
brightness = cv2.convertScaleAbs(asli, alpha=1.0, beta=geser)

# b) Contrast stretching
#    Meregangkan intensitas dari rentang sempit (persentil 2%-98%) ke 0-255.
rendah, tinggi = np.percentile(asli, (2, 98))
stretch = (asli.astype(float) - rendah) / (tinggi - rendah) * 255
stretching = np.clip(stretch, 0, 255).astype(np.uint8)

# c) Histogram equalization
#    Meratakan distribusi intensitas di seluruh rentang 0-255.
equalization = cv2.equalizeHist(asli)

# Kumpulkan semua hasil dalam satu daftar agar mudah diproses
hasil = {
    "Sebelum (asli)": asli,
    f"Brightness ({geser:+d})": brightness,
    "Contrast Stretching": stretching,
    "Histogram Equalization": equalization,
}


# ===============================================================
# LANGKAH 3: Hitung nilai untuk membandingkan kualitas
# ===============================================================
def hitung_nilai(gambar_gray):
    """Mengembalikan (kontras, persen piksel hitam/putih pekat)."""
    kontras = gambar_gray.std()                                   # makin besar = kontras makin tinggi
    pekat = ((gambar_gray <= 5) | (gambar_gray >= 250)).mean() * 100  # piksel 'terbakar'
    return kontras, pekat


print("\n=== PERBANDINGAN ===")
print(f"{'Metode':<26}{'Rata-rata':>10}{'Kontras':>10}{'% Pekat':>10}")
nilai = {}
for nama, img in hasil.items():
    kontras, pekat = hitung_nilai(img)
    nilai[nama] = (kontras, pekat)
    print(f"{nama:<26}{img.mean():>10.1f}{kontras:>10.1f}{pekat:>10.1f}")

# Saran metode terbaik: kontras tertinggi, tetapi tidak terlalu banyak piksel pekat
kandidat = {n: v for n, v in nilai.items() if n != "Sebelum (asli)" and v[1] < 10}
if not kandidat:
    kandidat = {n: v for n, v in nilai.items() if n != "Sebelum (asli)"}
terbaik = max(kandidat, key=lambda n: kandidat[n][0])
print(f"\nMetode yang disarankan untuk OCR: {terbaik}")


# ===============================================================
# LANGKAH 4 (opsional): OCR
# ===============================================================
try:
    import pytesseract
    print("\n=== HASIL OCR ===")
    for nama, img in hasil.items():
        teks = pytesseract.image_to_string(img, config="--psm 7").strip()
        print(f"{nama:<26}: {teks}")
except Exception:
    print("\n(OCR dilewati: pytesseract / Tesseract belum terpasang)")


# ===============================================================
# LANGKAH 5: Tampilkan citra dan histogram
# ===============================================================
plt.figure(figsize=(16, 7))
for i, (nama, img) in enumerate(hasil.items()):
    # Baris atas: citra
    plt.subplot(2, 4, i + 1)
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.title(nama)
    plt.axis("off")

    # Baris bawah: histogram
    plt.subplot(2, 4, i + 5)
    plt.hist(img.ravel(), bins=256, range=(0, 255), color="gray")
    plt.title(f"Histogram (kontras={nilai[nama][0]:.0f})")
    plt.xlabel("Intensitas")
    plt.ylabel("Jumlah piksel")

plt.suptitle(f"Enhancement Nomor Ijazah - disarankan: {terbaik}")
plt.tight_layout()
plt.savefig("hasil_tugas4.png", dpi=110)
print("\nGambar perbandingan disimpan: hasil_tugas4.png")
plt.show()
