def urutkan_sudut(pts):
    # Urutkan: Kiri-Atas, Kanan-Atas, Kanan-Bawah, Kiri-Bawah
    pts = pts.reshape((4, 2))
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-Left
    rect[2] = pts[np.argmax(s)]  # Bottom-Right
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-Right
    rect[3] = pts[np.argmax(diff)]  # Bottom-Left
    
    return rect

def warp_dokumen(img, pts):
    rect = urutkan_sudut(pts)
    (tl, tr, br, bl) = rect
    
    # Hitung ukuran dokumen baru
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    # Transformasi perspektif
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    
    return warped

# Tampilkan hasil warp untuk foto yang BERHASIL terdeteksi
plt.figure(figsize=(15, 10))
plot_idx = 1

for filename in foto_list:
    img_path = f'foto_uji/{filename}'
    img, cnt = deteksi_dokumen(img_path)
    
    if cnt is not None and plot_idx <= 6:  # Tampilkan maksimal 6 hasil
        warped = warp_dokumen(img, cnt)
        
        plt.subplot(2, 3, plot_idx)
        plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        plt.title(f"Dokumen Terwarp\n{filename}", fontsize=10)
        plt.axis('off')
        plot_idx += 1

plt.tight_layout()
plt.show()

print("\n📊 CHECKPOINT 2 - ANALISIS:")
print(f"   Total foto diuji: {len(foto_list)}")
print(f"   Berhasil terdeteksi: {sum(1 for f in foto_list if deteksi_dokumen(f'foto_uji/{f}')[1] is not None)}")
print(f"   Gagal terdeteksi: {sum(1 for f in foto_list if deteksi_dokumen(f'foto_uji/{f}')[1] is None)}")
print("\n💡 Penyebab kegagalan biasanya:")
print("   1. Bayangan tajam menutupi tepi dokumen")
print("   2. Sudut kamera terlalu miring (extreme tilt)")
print("   3. Kertas kusut/lipatan membuat tepi tidak beraturan")
print("   4. Warna latar meja mirip dengan warna kertas")
