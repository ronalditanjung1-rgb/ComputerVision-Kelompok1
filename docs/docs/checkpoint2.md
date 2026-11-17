import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import urllib.request
import os
from google.colab import files
import warnings
warnings.filterwarnings('ignore')

# Setup tampilan grafik agar jelas
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['figure.dpi'] = 120

print("="*70)
print(" CHECKPOINT 2: PERSPECTIVE TRANSFORM PIPELINE (SIAP JALAN)")
print("="*70)

# 1. Coba download gambar 
sample_url = ![Uploading 20261117_181634.jpg…]

sample_path = "sample_document.jpg"

try:
    urllib.request.urlretrieve(sample_url, sample_path)
    print("✅ Berhasil mengunduh GAMBAR CONTOH otomatis.")
except:
    print("⚠️ Gagal mengunduh gambar contoh. Silakan upload manual.")

# 2. Beri opsi upload foto Anda sendiri (Opsional)
print("\n📂 Ingin menggunakan foto sendiri? Upload sekarang (Tahan Ctrl/Cmd untuk pilih banyak).")
print("   (Jika tidak, sistem akan otomatis memakai gambar contoh di atas)\n")
try:
    uploaded = files.upload()
    image_paths = list(uploaded.keys())
except:
    image_paths = []

# Jika tidak ada yang di-upload, pakai gambar contoh
if not image_paths:
    image_paths = [sample_path]
    print(f"\n💡 Menggunakan gambar contoh: {sample_path}")
else:
    print(f"\n✅ Berhasil memuat {len(image_paths)} gambar dari upload And


#lanjutan dari diatas
# ============================================================
# BAGIAN A: DEFINISI FUNGSI
# ============================================================
def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.bilateralFilter(gray, 9, 75, 75)

def detect_edges(denoised):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    edges = cv2.Canny(enhanced, 50, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

def find_document_corners(closed_edges):
    contours, _ = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # Top-Left
    rect[2] = pts[np.argmax(s)]   # Bottom-Right
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # Top-Right
    rect[3] = pts[np.argmax(d)]   # Bottom-Left
    return rect

def warp_document(image, pts):
    rect = order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect
    w1, w2 = np.linalg.norm(br - bl), np.linalg.norm(tr - tl)
    h1, h2 = np.linalg.norm(tr - br), np.linalg.norm(tl - bl)
    maxW, maxH = int(max(w1, w2)), int(max(h1, h2))
    dst = np.array([[0,0], [maxW-1,0], [maxW-1,maxH-1], [0,maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH)), rect, M

def calculate_error(detected, gt):
    if detected is None or gt is None or len(gt) < 4:
        return np.inf
    return np.mean([np.linalg.norm(detected[i] - gt[i]) for i in range(4)])

# ============================================================
# BAGIAN B: EKSEKUSI & VISUALISASI 
# ============================================================
print("\n" + "="*70)
print("🔄 MEMPROSES GAMBAR & MENAMPILKAN HASIL...")
print("="*70)

# Ground Truth Manual (Contoh format, bisa Anda isi nanti)
ground_truth = {
    # "nama_file.jpg": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
}

results_table = []

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"⚠️ Gagal membaca: {path}")
        continue

    print(f"\n📷 Memproses: {path}")

    # Jalankan 4 Tahapan
    denoised = preprocess(img)
    closed = detect_edges(denoised)
    corners = find_document_corners(closed)

    success = corners is not None
    warped, rect, M = (None, None, None)

    if success:
        warped, rect, M = warp_document(img, corners)
        print("   ✅ BERHASIL menemukan 4 sudut & melakukan Warp.")
    else:
        print("   ❌ GAGAL menemukan 4 sudut (Coba foto dengan latar lebih kontras).")

    # --- VISUALISASI LANGSUNG DI OUTPUT COLAB ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Hasil Checkpoint 2: {path}', fontsize=16, fontweight='bold', y=0.98)

    # 1. Original
    axes[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title('1. Original Image', fontweight='bold'); axes[0,0].axis('off')

    # 2. Pre-processed
    axes[0,1].imshow(denoised, cmap='gray')
    axes[0,1].set_title('2. Pre-processed (Bilateral)', fontweight='bold'); axes[0,1].axis('off')

    # 3. Edges + Morph Close
    axes[0,2].imshow(closed, cmap='gray')
    axes[0,2].set_title('3. Edges + Morph Close', fontweight='bold'); axes[0,2].axis('off')

    # 4. Detected Corners
    img_pts = img.copy()
    if success:
        for i, pt in enumerate(corners):
            cv2.circle(img_pts, tuple(pt.astype(int)), 12, (0, 0, 255), -1)
            cv2.putText(img_pts, f"P{i+1}", (int(pt[0])+15, int(pt[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    axes[1,0].imshow(cv2.cvtColor(img_pts, cv2.COLOR_BGR2RGB))
    axes[1,0].set_title(f'4. Detected Corners', fontweight='bold'); axes[1,0].axis('off')

    # 5. Warped Result
    if success:
        axes[1,1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        axes[1,1].set_title('5. Warped (Top-Down View)', fontweight='bold')
    else:
        axes[1,1].text(0.5, 0.5, '❌ GAGAL WARP', transform=axes[1,1].transAxes,
                      ha='center', fontsize=20, color='red', fontweight='bold')
    axes[1,1].axis('off')

    # 6. Homography Matrix Info
    axes[1,2].axis('off')
    axes[1,2].set_title('6. Homography Matrix (M)', fontweight='bold')
    if success:
        text = "M =\n"
        for row in M:
            text += "[" + "  ".join([f"{v:7.2f}" for v in row]) + "]\n"
        axes[1,2].text(0.05, 0.5, text, transform=axes[1,2].transAxes,
                      fontsize=11, family='monospace', color='darkblue',
                      bbox=dict(boxstyle='round', facecolor='#f0f8ff', edgecolor='blue'))
    else:
        axes[1,2].text(0.5, 0.5, 'NO MATRIX', transform=axes[1,2].transAxes,
                      ha='center', fontsize=16, color='red')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()  # <-- INI YANG MEMUNCULKAN GAMBAR LANGSUNG DI COLAB

    # Simpan data untuk tabel
    err = calculate_error(corners, np.array(ground_truth.get(path, []))) if success else np.inf
    results_table.append({
        'File': os.path.basename(path),
        'Status': '✅ Berhasil' if success else '❌ Gagal',
        'Galat (px)': f"{err:.2f}" if err != np.inf else '-'
    })

# ============================================================
# BAGIAN C: TABEL EVALUASI 
# ============================================================
print("\n" + "="*70)
print("📊 TABEL EVALUASI CHECKPOINT 2")
print("="*70)
df = pd.DataFrame(results_table)
print(df.to_string(index=False))

success_rate = sum(1 for r in results_table if 'Berhasil' in r['Status']) / len(results_table) * 100
print(f"\n📈 TINGKAT KEBERHASILAN DETEKSI: {success_rate:.1f}%")
print("="*70)
