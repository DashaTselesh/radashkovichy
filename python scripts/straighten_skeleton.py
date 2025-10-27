import cv2
import numpy as np
from skimage.morphology import skeletonize, medial_axis
from scipy import ndimage
import json

# ====================
# ПАРАМЕТРЫ
# ====================
INPUT_IMAGE = 'map_1891_transparent.png'
OUTPUT_SKELETON = 'map_1891_skeleton_straight.png'

# Метод выравнивания
STRAIGHTEN_METHOD = 'bilateral'  # 'bilateral', 'median', 'none'
BILATERAL_D = 9  # Диаметр окрестности (5, 7, 9, 11)
BILATERAL_SIGMA_COLOR = 75  # Фильтр цвета
BILATERAL_SIGMA_SPACE = 75  # Фильтр пространства

# Удаление изолированных точек
MIN_COMPONENT_SIZE = 10

print("=" * 60)
print("ВЫРАВНИВАНИЕ СКЕЛЕТА (Сохранение всех линий)")
print("=" * 60)
print(f"Входной файл: {INPUT_IMAGE}")
print(f"Выходной файл: {OUTPUT_SKELETON}")
print(f"Метод: {STRAIGHTEN_METHOD}")
if STRAIGHTEN_METHOD == 'bilateral':
    print(f"Bilateral d={BILATERAL_D}, sigma={BILATERAL_SIGMA_COLOR}")

# ====================
# ШАГ 1: Загрузка
# ====================
print("\n[1/5] Загрузка изображения...")
img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(f"Файл {INPUT_IMAGE} не найден!")

height, width = img.shape[:2]
print(f"Размер: {width}x{height} пикселей")

alpha_mask = None
if len(img.shape) == 3 and img.shape[2] == 4:
    print("Обнаружен альфа-канал")
    alpha = img[:, :, 3]
    rgb = img[:, :, :3]
    alpha_mask = alpha > 50
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
else:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

# ====================
# ШАГ 2: Бинаризация
# ====================
print("\n[2/5] Бинаризация...")
_, binary_white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

if alpha_mask is not None:
    binary_white[~alpha_mask] = 0

binary_black = cv2.bitwise_not(binary_white)
if alpha_mask is not None:
    binary_black[~alpha_mask] = 0

cv2.imwrite('debug_01_boundaries.png', binary_black)
print("→ debug_01_boundaries.png (исходные границы)")

# ====================
# ШАГ 3: Выравнивание границ
# ====================
print(f"\n[3/5] Выравнивание границ (метод: {STRAIGHTEN_METHOD})...")

if STRAIGHTEN_METHOD == 'bilateral':
    # Bilateral filter сохраняет края, но сглаживает
    smoothed = cv2.bilateralFilter(binary_black, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)

    cv2.imwrite('debug_02_bilateral.png', smoothed)
    print(f"→ debug_02_bilateral.png (после bilateral filter)")

    # Бинаризация
    _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)

elif STRAIGHTEN_METHOD == 'median':
    # Медианный фильтр
    smoothed = cv2.medianBlur(binary_black, 5)
    cv2.imwrite('debug_02_median.png', smoothed)
    print(f"→ debug_02_median.png (после median filter)")

else:  # 'none'
    smoothed = binary_black
    print("Выравнивание пропущено")

cv2.imwrite('debug_03_smoothed_binary.png', smoothed)
print("→ debug_03_smoothed_binary.png (бинаризованные сглаженные границы)")

# ====================
# ШАГ 4: Скелетизация
# ====================
print("\n[4/5] Скелетизация...")

skeleton_bool = skeletonize(smoothed > 0)
skeleton = (skeleton_bool * 255).astype(np.uint8)

print(f"Пикселей в скелете: {np.sum(skeleton > 0)}")

cv2.imwrite('debug_04_skeleton.png', skeleton)
print("→ debug_04_skeleton.png (скелет)")

# ====================
# ШАГ 5: Удаление изолированных точек
# ====================
print(f"\n[5/5] Удаление изолированных точек (мин. размер: {MIN_COMPONENT_SIZE}px)...")

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton, connectivity=8)

skeleton_cleaned = np.zeros_like(skeleton)
removed_count = 0

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if area >= MIN_COMPONENT_SIZE:
        skeleton_cleaned[labels == i] = 255
    else:
        removed_count += 1

skeleton = skeleton_cleaned
print(f"Удалено маленьких компонент: {removed_count}")

# ====================
# Сохранение
# ====================
output = cv2.bitwise_not(skeleton)
if alpha_mask is not None:
    output[~alpha_mask] = 255

cv2.imwrite(OUTPUT_SKELETON, output)
print(f"\n→ {OUTPUT_SKELETON}")
print("Формат: черные линии на белом фоне")

print("\n" + "=" * 60)
print("ГОТОВО!")
print("=" * 60)
print(f"\nДля QGIS:")
print(f"  Raster → Conversion → Polygonize")
print(f"  Файл: {OUTPUT_SKELETON}")
print("\nЕсли нужно еще прямее:")
print(f"  Увеличьте BILATERAL_D до 11-15")
print("=" * 60)
