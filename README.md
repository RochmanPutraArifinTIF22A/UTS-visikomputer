# UTS-visikomputer
# Install library yang diperlukan (jika belum ada)
!pip install opencv-python matplotlib

# Import library
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from IPython.display import Image, display

# Upload gambar iris
uploaded = files.upload()

# Ambil nama file
filename = list(uploaded.keys())[0]

# Baca gambar
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Terapkan Otsu Thresholding
# Otsu otomatis menentukan nilai threshold optimal
ret, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Tampilkan hasil
plt.figure(figsize=(12,4))

plt.subplot(1, 3, 1)
plt.title('Citra Asli')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Grayscale')
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title(f'Otsu Thresholding\nThreshold: {ret:.2f}')
plt.imshow(otsu, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
