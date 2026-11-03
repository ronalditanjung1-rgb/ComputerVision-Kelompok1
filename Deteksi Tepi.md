# Haluskan gambar terlebih dahulu untuk meredam bintik derau (noise)
blur = cv2.GaussianBlur(abu, (5, 5), 0)


# Bandingkan deteksi tepi dengan ambang batas ketat vs. ambang batas sensitif
tepi_a = cv2.Canny(blur, 100, 200)
tepi_b = cv2.Canny(blur, 30, 80)
cv2_imshow(tepi_a)
cv2_imshow(tepi_b)
