import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import os

# Upload foto dari komputer
print(" Silakan upload foto dokumen Anda (bisa pilih banyak sekaligus)")
uploaded = files.upload()

# Simpan ke folder 'foto_uji'
os.makedirs('foto_uji', exist_ok=True)
for filename in uploaded.keys():
    with open(f'foto_uji/{filename}', 'wb') as f:
        f.write(uploaded[filename])

print(f"\n {len(uploaded)} foto berhasil diupload!")




def deteksi_dokumen(image_path):
    # Baca gambar
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur + Canny Edge
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    # Cari kontur
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Cari kontur dengan 4 sudut
    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    return img, screenCnt

# Tampilkan hasil untuk semua foto yang diupload
foto_list = os.listdir('foto_uji')
plt.figure(figsize=(15, 10))

for i, filename in enumerate(foto_list):
    img_path = f'foto_uji/{filename}'
    img, cnt = deteksi_dokumen(img_path)

    plt.subplot(2, 3, i+1)
    if cnt is not None:
        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)
        plt.title(f" BERHASIL\n{filename}", fontsize=10)
    else:
        plt.title(f" GAGAL\n{filename}", fontsize=10)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

plt.tight_layout()
plt.show()
print("\n Analisis: Foto yang BERHASIL menunjukkan 4 sudut dokumen terdeteksi (kotak hijau)")
print("   Foto yang GAGAL biasanya karena: bayangan, sudut miring, atau latar mirip kertas")
