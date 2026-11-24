import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import urllib.request
import os
from google.colab import files
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (18, 12)
plt.rcParams['figure.dpi'] = 120

print("=" * 75)
print(" CHECKPOINT 2: PERSPECTIVE TRANSFORM PIPELINE (LENGKAP POIN 5 - 13)")
print("=" * 75)


# ============================================================================
# 1. INPUT GAMBAR (UPLOAD ATAU CONTOH)
# ============================================================================
sample_path = "sample_document.jpg"
if not os.path.exists(sample_path):
    # Membuat gambar sintetis cadangan jika URL tidak tersedia
    canvas = np.ones((720, 960, 3), dtype=np.uint8) * 70
    cv2.rectangle(canvas, (180, 120), (780, 600), (235, 235, 235), -1)
    cv2.putText(canvas, "DOKUMEN CONTOH - UJI HOMOGRAFI", (220, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
    cv2.imwrite(sample_path, canvas)

print("\n📂 Upload foto dokumen Anda sekarang (bisa pilih lebih dari satu):")
try:
    uploaded = files.upload()
    image_paths = list(uploaded.keys())
except Exception:
    image_paths = []

if not image_paths:
    image_paths = [sample_path]
    print(f"\n💡 Menggunakan gambar contoh: {sample_path}")
else:
    print(f"\n✅ Berhasil memuat {len(image_paths)} gambar dari upload Anda.")

# Ground Truth opsional (jika ingin hitung galat piksel)
ground_truth = {
    # "nama_file.jpg": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
}

# ============================================================================
# BAGIAN A: FUNGSI PRA-PROSES, TEPI, CANNY MODIFIKASI, & HOMOGRAFI
# ============================================================================

def preprocess_anti_bias(image):
    """
    Poin 5 & 8:
    - Konversi ke grayscale & normalisasi pencahayaan (LAB L-channel / CLAHE)
      untuk mengurangi bias pencahayaan tidak merata sebelum deteksi tepi.
    - Menghasilkan versi Bilateral Filter dan Gaussian Blur untuk perbandingan.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalisasi pencahayaan agar bayangan/kilau tidak mendominasi gradien
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe_l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_channel)
    norm_lab = cv2.merge((clahe_l, a_channel, b_channel))
    gray_norm = cv2.cvtColor(norm_lab, cv2.COLOR_LAB2BGR)
    gray_norm = cv2.cvtColor(gray_norm, cv2.COLOR_BGR2GRAY)

    # Noise reduction: Bilateral (jaga tepi tajam) & Gaussian Blur (halus merata)
    bilateral = cv2.bilateralFilter(gray_norm, d=9, sigmaColor=75, sigmaSpace=75)
    gaussian = cv2.GaussianBlur(gray_norm, (7, 7), sigmaX=1.5)

    return gray, gray_norm, bilateral, gaussian


def detect_edge_polarity(gray_img):
    """
    Poin 6:
    Membedakan arah transisi intensitas tepi menggunakan gradien Sobel (dx, dy):
    - Tepi Terang -> Gelap (misal kertas putih ke latar/tinta gelap): gradien negatif
    - Tepi Gelap -> Terang (misal latar gelap ke kertas putih): gradien positif
    """
    gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    # Masker piksel terang vs gelap di sekitar tepi
    # Gradien positif pada arah x/y menandakan kenaikan intensitas (menuju putih/terang)
    # Gradien negatif menandakan penurunan intensitas (menuju hitam/gelap)
    to_light = np.clip((gx + gy), 0, None)
    to_dark = np.clip(-(gx + gy), 0, None)

    to_light_u8 = cv2.normalize(to_light, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    to_dark_u8 = cv2.normalize(to_dark, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Visualisasi RGB: Merah = transisi ke terang (putih), Biru/Cyan = transisi ke gelap (hitam)
    polarity_vis = np.zeros((gray_img.shape[0], gray_img.shape[1], 3), dtype=np.uint8)
    polarity_vis[:, :, 0] = to_light_u8   # R: transisi gelap -> terang
    polarity_vis[:, :, 2] = to_dark_u8    # B: transisi terang -> gelap
    polarity_vis[:, :, 1] = (mag > 40).astype(np.uint8) * 60

    return to_light_u8, to_dark_u8, polarity_vis


def auto_canny_thresholds(img, sigma=0.33):
    """Poin 12: Menghitung threshold Canny adaptif berbasis median intensitas."""
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    if upper <= lower:
        upper = min(255, lower + 50)
    return lower, upper


def find_corners_from_edges(closed_edges, img_shape):
    """Mencari kontur segi empat terbesar dengan luas minimal yang wajar."""
    h, w = img_shape[:2]
    min_area = 0.05 * (h * w)
    contours, _ = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours[:15]:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2)
    return None


def modified_canny_pipeline(bilateral, gaussian, img_shape):
    """
    Poin 8, 9, 12, 13:
    Menjalankan Canny awal (50, 200), lalu jika gagal mencari 4 sudut dokumen,
    mencoba Canny termodifikasi (Auto-Canny pada Bilateral, Gaussian Blur, dan threshold longgar).
    Mencatat threshold yang gagal, nilai yang dicoba, root cause, dan solusi.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Percobaan 1: Canny standar bawaan (50, 200) pada Bilateral
    low_init, high_init = 50, 200
    edges_init = cv2.Canny(bilateral, low_init, high_init)
    closed_init = cv2.morphologyEx(edges_init, cv2.MORPH_CLOSE, kernel)
    corners_init = find_corners_from_edges(closed_init, img_shape)

    logs = []
    if corners_init is not None:
        return {
            "success": True,
            "corners": corners_init,
            "edges": edges_init,
            "closed": closed_init,
            "used_low": low_init,
            "used_high": high_init,
            "mode_used": "Canny Standar (Bilateral)",
            "failed_threshold": None,
            "fail_caption": "",
            "root_cause": "Tidak ada kegagalan pada percobaan awal.",
            "solution": "Threshold (50, 200) sudah cukup memisahkan tepi dokumen.",
            "logs": logs
        }

    # Catat kegagalan awal
    fail_low, fail_high = low_init, high_init
    logs.append(f"Percobaan 1 GAGAL di threshold ({fail_low}, {fail_high}) pada Bilateral.")

    # Kandidat modifikasi Canny (Poin 8 & 12)
    auto_low, auto_high = auto_canny_thresholds(bilateral, sigma=0.33)
    gauss_low, gauss_high = auto_canny_thresholds(gaussian, sigma=0.40)

    candidates = [
        ("Canny Modifikasi #1 (Auto-Canny Bilateral)", bilateral, auto_low, auto_high),
        ("Canny Modifikasi #2 (Gaussian Blur + Auto-Canny)", gaussian, gauss_low, gauss_high),
        ("Canny Modifikasi #3 (Gaussian Blur + Low Threshold 25, 100)", gaussian, 25, 100),
    ]

    last_edges = edges_init
    last_closed = closed_init

    for name, src_img, lo, hi in candidates:
        e = cv2.Canny(src_img, lo, hi)
        c = cv2.morphologyEx(e, cv2.MORPH_CLOSE, kernel)
        pts = find_corners_from_edges(c, img_shape)
        last_edges, last_closed = e, c
        if pts is not None:
            logs.append(f"BERHASIL dengan {name} pada threshold ({lo}, {hi}).")
            return {
                "success": True,
                "corners": pts,
                "edges": e,
                "closed": c,
                "used_low": lo,
                "used_high": hi,
                "mode_used": name,
                "failed_threshold": f"({fail_low}, {fail_high})",
                "fail_caption": f"GAGAL pada threshold awal ({fail_low}, {fail_high}); pulih via {name} [{lo}, {hi}]",
                "root_cause": (
                    f"Root Cause: Pada threshold ({fail_low}, {fail_high}), gradien tepi dokumen "
                    f"lebih lemah dari high_threshold={fail_high} atau kontur terputus sehingga approxPolyDP "
                    f"tidak membentuk 4 titik konveks."
                ),
                "solution": (
                    f"Solusi: Gunakan {name} dengan threshold ({lo}, {hi}) serta normalisasi pencahayaan/blur "
                    f"untuk menyambungkan tepi yang terputus."
                ),
                "logs": logs
            }
        else:
            logs.append(f"{name} pada ({lo}, {hi}) juga belum menemukan 4 sudut.")

    # Jika seluruh kandidat masih gagal
    fail_caption = (
        f"GAGAL DETEKSI 4 SUDUT | Threshold gagal: awal ({fail_low}, {fail_high}), "
        f"auto ({auto_low}, {auto_high}), gauss ({gauss_low}, {gauss_high})"
    )
    root_cause = (
        "Root Cause: (1) Kontras tepi dokumen vs latar belakang terlalu rendah atau terhalang noise/bayangan, "
        "(2) Sudut dokumen terpotong/melengkung sehingga bukan poligon 4 sisi tertutup, atau "
        "(3) Objek lain di latar memiliki kontur segi empat lebih besar."
    )
    solution = (
        "Solusi: (1) Foto ulang di latar kontras dengan pencahayaan merata, (2) Gunakan adaptive threshold / "
        "deteksi garis HoughLinesP sebagai alternatif kontur, (3) Anotasi 4 sudut manual jika perlu."
    )
    return {
        "success": False,
        "corners": None,
        "edges": last_edges,
        "closed": last_closed,
        "used_low": gauss_low,
        "used_high": gauss_high,
        "mode_used": "Semua Mode Gagal",
        "failed_threshold": f"({fail_low}, {fail_high}) -> ({gauss_low}, {gauss_high})",
        "fail_caption": fail_caption,
        "root_cause": root_cause,
        "solution": solution,
        "logs": logs
    }


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
    maxW, maxH = max(int(max(w1, w2)), 100), max(int(max(h1, h2)), 100)
    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH))
    return warped, rect, dst, M


def draw_guide_lines(img_bgr, step_px=80):
    """Poin 11: Menambahkan garis panduan (grid & garis tengah) untuk inspeksi kelurusan."""
    guided = img_bgr.copy()
    h, w = guided.shape[:2]
    for x in range(step_px, w, step_px):
        cv2.line(guided, (x, 0), (x, h), (0, 255, 255), 1)
    for y in range(step_px, h, step_px):
        cv2.line(guided, (0, y), (w, y), (0, 255, 255), 1)
    # Garis sumbu tengah merah muda/magenta
    cv2.line(guided, (w // 2, 0), (w // 2, h), (255, 0, 255), 2)
    cv2.line(guided, (0, h // 2), (w, h // 2), (255, 0, 255), 2)
    return guided


def calculate_error(detected, gt):
    if detected is None or gt is None or len(gt) < 4:
        return np.inf
    det_ord = order_points(np.array(detected, dtype="float32"))
    gt_ord = order_points(np.array(gt, dtype="float32"))
    return np.mean([np.linalg.norm(det_ord[i] - gt_ord[i]) for i in range(4)])


# ============================================================================
# BAGIAN B: EKSEKUSI, VISUALISASI, MATRIKS & GRAFIK HOMOGRAFI
# ============================================================================
print("\n" + "=" * 75)
print("🔄 MEMPROSES GAMBAR & MENAMPILKAN HASIL...")
print("=" * 75)

results_table = []
homography_records = []

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"⚠️ Gagal membaca: {path}")
        continue

    print(f"\n📷 Memproses: {path}")

    # Pra-proses anti-bias + Bilateral + Gaussian Blur
    gray, gray_norm, bilateral, gaussian = preprocess_anti_bias(img)

    # Polaritas tepi: arah gelap vs terang
    _, _, polarity_vis = detect_edge_polarity(gray_norm)

    # Canny modifikasi & analisis kegagalan
    res = modified_canny_pipeline(bilateral, gaussian, img.shape)
    success = res["success"]
    corners = res["corners"]
    closed = res["closed"]

    warped, rect, dst_pts, M = (None, None, None, None)
    if success:
        warped, rect, dst_pts, M = warp_document(img, corners)
        print(f"   ✅ BERHASIL ({res['mode_used']}) | Threshold: ({res['used_low']}, {res['used_high']})")
    else:
        print(f"   ❌ {res['fail_caption']}")

    print(f"   🔍 {res['root_cause']}")
    print(f"   🛠️  {res['solution']}")

    # --- LAYOUT VISUALISASI (3x3) ---
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    status_tag = "BERHASIL" if success else "GAGAL"
    fig.suptitle(f"Checkpoint 2 [{status_tag}]: {path}", fontsize=16, fontweight='bold', y=0.98)

    # [0,0] Original + Garis Panduan
    orig_guide = draw_guide_lines(img, step_px=100)
    axes[0, 0].imshow(cv2.cvtColor(orig_guide, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Original + Garis Panduan (Poin 11)", fontweight='bold')
    axes[0, 0].axis('off')

    # [0,1] Pra-proses Anti-Bias (Bilateral)
    axes[0, 1].imshow(bilateral, cmap='gray')
    axes[0, 1].set_title("2. Pra-Proses Anti-Bias: LAB-CLAHE + Bilateral", fontweight='bold')
    axes[0, 1].axis('off')

    # [0,2] Gaussian Blur
    axes[0, 2].imshow(gaussian, cmap='gray')
    axes[0, 2].set_title("3. Gaussian Blur (7x7, sigma=1.5)", fontweight='bold')
    axes[0, 2].axis('off')

    # [1,0] Polaritas Tepi (Gelap vs Terang)
    axes[1, 0].imshow(cv2.cvtColor(polarity_vis, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("4. Polaritas Tepi: Merah=Ke Terang, Biru=Ke Gelap", fontweight='bold')
    axes[1, 0].axis('off')

    # [1,1] Canny Modifikasi + Morph Close + Caption Gagal/Threshold
    axes[1, 1].imshow(closed, cmap='gray')
    title_canny = (
        f"5. Canny ({res['used_low']}, {res['used_high']}) + Close\n"
        f"{res['fail_caption'] if res['fail_caption'] else 'Status: OK'}"
    )
    axes[1, 1].set_title(title_canny, fontweight='bold',
                         color='darkred' if not success or res['fail_caption'] else 'darkgreen',
                         fontsize=10)
    axes[1, 1].axis('off')

    # [1,2] Deteksi 4 Sudut + Garis Panduan
    img_pts = draw_guide_lines(img, step_px=100)
    if success:
        poly_pts = rect.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_pts, [poly_pts], True, (0, 255, 0), 3)
        for i, pt in enumerate(rect):
            cv2.circle(img_pts, tuple(pt.astype(int)), 11, (0, 0, 255), -1)
            cv2.putText(img_pts, f"P{i+1}({int(pt[0])},{int(pt[1])})",
                        (int(pt[0]) + 10, int(pt[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    else:
        cv2.putText(img_pts, "GAGAL DETEKSI 4 SUDUT", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    axes[1, 2].imshow(cv2.cvtColor(img_pts, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("6. Sudut Terdeteksi & Garis Panduan", fontweight='bold')
    axes[1, 2].axis('off')

    # [2,0] Hasil Warp + Garis Panduan
    if success:
        warped_guide = draw_guide_lines(warped, step_px=80)
        axes[2, 0].imshow(cv2.cvtColor(warped_guide, cv2.COLOR_BGR2RGB))
        axes[2, 0].set_title(f"7. Warped ({warped.shape[1]}x{warped.shape[0]}px) + Panduan", fontweight='bold')
    else:
        axes[2, 0].text(0.5, 0.5, f"GAGAL WARP\n{res['fail_caption'][:75]}",
                      transform=axes[2, 0].transAxes, ha='center', va='center',
                      fontsize=11, color='red', fontweight='bold')
    axes[2, 0].axis('off')

    # [2,1] Matriks Homografi M (3x3)
    axes[2, 1].axis('off')
    axes[2, 1].set_title("8. Matriks Homografi (M 3x3)", fontweight='bold')
    if success and M is not None:
        m_text = "Matriks Homografi M =\n"
        for r_idx in range(3):
            m_text += f"[ {M[r_idx,0]:9.4f}  {M[r_idx,1]:9.4f}  {M[r_idx,2]:9.2f} ]\n"
        axes[2, 1].text(0.03, 0.5, m_text, transform=axes[2, 1].transAxes,
                      fontsize=10.5, family='monospace', color='#0b2545', va='center',
                      bbox=dict(boxstyle='round', facecolor='#eef6ff', edgecolor='#1d4ed8', lw=1.5))
    else:
        fail_box = (
            f"NO HOMOGRAPHY MATRIX\n"
            f"Threshold: {res['failed_threshold']}\n"
            f"Root Cause: {res['root_cause'][:120]}..."
        )
        axes[2, 1].text(0.03, 0.5, fail_box, transform=axes[2, 1].transAxes,
                      fontsize=9.5, color='darkred', va='center',
                      bbox=dict(boxstyle='round', facecolor='#fff5f5', edgecolor='red', lw=1.5))

    # [2,2] Grafik Nilai Elemen Matriks Homografi (Poin 7 & 10)
    if success and M is not None:
        labels_m = [f"m{i+1}{j+1}" for i in range(3) for j in range(3)]
        vals_m = M.flatten()
        colors_m = ['#2563eb' if v >= 0 else '#dc2626' for v in vals_m]
        bars = axes[2, 2].bar(labels_m, vals_m, color=colors_m, edgecolor='black', alpha=0.85)
        axes[2, 2].axhline(0, color='black', linewidth=1)
        axes[2, 2].set_title("9. Grafik Nilai Elemen Matriks M", fontweight='bold')
        axes[2, 2].set_ylabel("Nilai Elemen")
        axes[2, 2].tick_params(axis='x', rotation=45)
        axes[2, 2].grid(axis='y', linestyle='--', alpha=0.5)
        for bar, val in zip(bars, vals_m):
            axes[2, 2].text(bar.get_x() + bar.get_width() / 2, val,
                            f"{val:.1f}", ha='center',
                            va='bottom' if val >= 0 else 'top', fontsize=8)
    else:
        axes[2, 2].text(0.5, 0.5, "Grafik Homografi Kosong\n(Deteksi Gagal)",
                      transform=axes[2, 2].transAxes, ha='center', va='center',
                      fontsize=12, color='red', fontweight='bold')
        axes[2, 2].axis('off')

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.show()

    # Catatan evaluasi & tabel homografi
    err = calculate_error(corners, np.array(ground_truth.get(path, []))) if success else np.inf
    results_table.append({
        'File': os.path.basename(path),
        'Status': '✅ Berhasil' if success else '❌ Gagal',
        'Mode / Threshold': f"{res['mode_used']} ({res['used_low']}, {res['used_high']})",
        'Threshold Gagal': res['failed_threshold'] if res['failed_threshold'] else '-',
        'Galat (px)': f"{err:.2f}" if err != np.inf else '-',
        'Root Cause & Solusi': f"{res['root_cause']} | {res['solution']}"
    })

    if success and M is not None:
        homography_records.append({
            'File': os.path.basename(path),
            'm11': round(float(M[0, 0]), 4),
            'm12': round(float(M[0, 1]), 4),
            'm13': round(float(M[0, 2]), 4),
            'm21': round(float(M[1, 0]), 4),
            'm22': round(float(M[1, 1]), 4),
            'm23': round(float(M[1, 2]), 4),
            'm31': round(float(M[2, 0]), 6),
            'm32': round(float(M[2, 1]), 6),
            'm33': round(float(M[2, 2]), 4),
        })

# ============================================================================
# BAGIAN C: TABEL EVALUASI, TABEL HOMOGRAFI, GRAFIK GABUNGAN & REFERENSI
# ============================================================================
print("\n" + "=" * 75)
print("📊 TABEL EVALUASI CHECKPOINT 2 (THRESHOLD, ROOT CAUSE & SOLUSI)")
print("=" * 75)
df_eval = pd.DataFrame(results_table)
print(df_eval[['File', 'Status', 'Mode / Threshold', 'Threshold Gagal', 'Galat (px)']].to_string(index=False))

print("\n🔎 DETAIL ROOT CAUSE & SOLUSI PER FILE:")
for row in results_table:
    print(f"• {row['File']} [{row['Status']}]: {row['Root Cause & Solusi']}")

print("\n" + "=" * 75)
print("🧮 TABEL MATRIKS HOMOGRAFI (M)")
print("=" * 75)
if homography_records:
    df_homo = pd.DataFrame(homography_records)
    print(df_homo.to_string(index=False))

    # Grafik perbandingan parameter homografi antar file berhasil
    fig_h, ax_h = plt.subplots(figsize=(14, 5))
    df_plot = df_homo.set_index('File')[['m11', 'm12', 'm13', 'm21', 'm22', 'm23', 'm33']]
    df_plot.T.plot(kind='bar', ax=ax_h, edgecolor='black', alpha=0.88)
    ax_h.set_title("Grafik Perbandingan Elemen Matriks Homografi (M) per Gambar", fontweight='bold')
    ax_h.set_xlabel("Elemen Matriks Homografi")
    ax_h.set_ylabel("Nilai")
    ax_h.axhline(0, color='black', linewidth=1)
    ax_h.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
else:
    print("Tidak ada matriks homografi yang terbentuk karena seluruh gambar gagal deteksi 4 sudut.")

success_rate = (sum(1 for r in results_table if 'Berhasil' in r['Status']) / max(len(results_table), 1)) * 100
print(f"\n📈 TINGKAT KEBERHASILAN DETEKSI: {success_rate:.1f}%")

