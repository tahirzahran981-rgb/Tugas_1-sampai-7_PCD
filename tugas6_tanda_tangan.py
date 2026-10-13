import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================================================
# PENGATURAN (ubah bagian ini saja)
# ===============================================================
FOLDER_INPUT = "images"      # folder berisi citra ijazah (ada TTD & tanpa TTD)
FOLDER_HASIL = "hasil_ttd"   # folder penyimpanan hasil

# Area tanda tangan: (x, y, lebar, tinggi) dalam piksel.
# None = pilih area dengan mouse pada gambar PERTAMA (drag lalu tekan ENTER).
# Area yang sama dipakai untuk semua gambar (asumsi tata letak ijazah sama).
AREA_TTD = None
SUDAH_DI_CROP = False        # True jika gambar di folder SUDAH berupa crop tanda tangan

THRESHOLD_GLOBAL = 127       # nilai ambang untuk global threshold
RASIO_MINIMAL = 1.0          # aturan: foreground >= 1% dari area -> PRESENT
SELISIH_TINTA = 40           # tinta harus minimal 40 tingkat lebih gelap dari kertas

os.makedirs(FOLDER_HASIL, exist_ok=True)


# ===============================================================
# FUNGSI: proses satu citra
# ===============================================================
def proses(crop_bgr):
    """Mengembalikan dictionary berisi semua tahap pemrosesan."""
    # 1. Grayscale lalu haluskan sedikit agar noise kecil berkurang
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 2. Thresholding. THRESH_BINARY_INV -> tinta (gelap) jadi PUTIH (foreground)
    _, global_th = cv2.threshold(gray, THRESHOLD_GLOBAL, 255, cv2.THRESH_BINARY_INV)
    _, otsu_th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive_th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 31, 10)

    # Pengaman: Otsu SELALU membagi gambar jadi 2 kelompok, bahkan pada kertas
    # kosong (noise dianggap tinta). Jadi cek dulu: adakah piksel yang cukup gelap?
    ada_tinta = (np.median(gray) - gray.min()) >= SELISIH_TINTA
    otsu_bersih = otsu_th if ada_tinta else np.zeros_like(otsu_th)

    # 3. Morphology pada hasil Otsu
    kernel_kecil = np.ones((2, 2), np.uint8)
    kernel_besar = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(otsu_bersih, cv2.MORPH_OPEN, kernel_kecil)       # hapus bintik
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_besar)      # sambung goresan

    # 4. Hitung karakteristik area hasil segmentasi
    jumlah_fg = int(np.count_nonzero(closing))      # jumlah piksel foreground
    rasio = jumlah_fg / closing.size * 100          # persentase terhadap seluruh area

    # 5. Aturan keputusan
    status = "SIGNATURE PRESENT" if rasio >= RASIO_MINIMAL else "SIGNATURE ABSENT"

    return {
        "gray": gray, "global": global_th, "otsu": otsu_th, "adaptive": adaptive_th,
        "opening": opening, "closing": closing,
        "fg_global": int(np.count_nonzero(global_th)),
        "fg_otsu": int(np.count_nonzero(otsu_th)),
        "fg_adaptive": int(np.count_nonzero(adaptive_th)),
        "fg_akhir": jumlah_fg, "rasio": rasio, "status": status,
    }


# ===============================================================
# PROGRAM UTAMA
# ===============================================================
daftar_file = sorted(f for f in os.listdir(FOLDER_INPUT)
                     if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
if not daftar_file:
    raise SystemExit(f"Tidak ada gambar di folder '{FOLDER_INPUT}'")

# Tentukan area crop (sekali saja, dari gambar pertama)
if not SUDAH_DI_CROP and AREA_TTD is None:
    pertama = cv2.imread(os.path.join(FOLDER_INPUT, daftar_file[0]))
    AREA_TTD = cv2.selectROI("Pilih area tanda tangan lalu tekan ENTER", pertama)
    cv2.destroyAllWindows()

print(f"\n{'No':<4}{'File':<28}{'FG Global':>10}{'FG Otsu':>9}{'FG Adapt':>9}"
      f"{'FG Akhir':>10}{'Rasio':>8}  Status")
print("-" * 95)

for nomor, nama_file in enumerate(daftar_file, start=1):
    gambar = cv2.imread(os.path.join(FOLDER_INPUT, nama_file))
    if SUDAH_DI_CROP:
        crop = gambar
    else:
        x, y, w, h = AREA_TTD
        crop = gambar[y:y + h, x:x + w]

    r = proses(crop)

    print(f"{nomor:<4}{nama_file[:26]:<28}{r['fg_global']:>10}{r['fg_otsu']:>9}"
          f"{r['fg_adaptive']:>9}{r['fg_akhir']:>10}{r['rasio']:>7.2f}%  {r['status']}")

    # ----- Tampilkan & simpan perbandingan -----
    tahap = [
        ("Crop (RGB)", cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
        ("Grayscale", r["gray"]),
        (f"Global (T={THRESHOLD_GLOBAL})\nFG={r['fg_global']}", r["global"]),
        (f"Otsu\nFG={r['fg_otsu']}", r["otsu"]),
        (f"Adaptive\nFG={r['fg_adaptive']}", r["adaptive"]),
        (f"Opening\nFG={int(np.count_nonzero(r['opening']))}", r["opening"]),
        (f"Closing (akhir)\nFG={r['fg_akhir']}", r["closing"]),
    ]
    plt.figure(figsize=(16, 4))
    for i, (judul, img) in enumerate(tahap):
        plt.subplot(1, 7, i + 1)
        plt.imshow(img, cmap=None if img.ndim == 3 else "gray")
        plt.title(judul, fontsize=9)
        plt.axis("off")
    plt.suptitle(f"{nama_file}  ->  {r['status']}  (rasio {r['rasio']:.2f}%)",
                 fontweight="bold", color="green" if "PRESENT" in r["status"] else "red")
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER_HASIL, f"ttd_{nomor}.png"), dpi=100)
    plt.close()

print(f"\nSelesai. Gambar perbandingan tersimpan di folder '{FOLDER_HASIL}'.")
