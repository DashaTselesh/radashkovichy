import cv2
import numpy as np
from skimage.morphology import skeletonize
import json

# ====================
# ПАРАМЕТРЫ
# ====================
INPUT_IMAGE = 'map_1891_transparent.png'
OUTPUT_SKELETON = 'map_1891_skeleton_smooth.png'

# Сглаживание САМОГО СКЕЛЕТА
SMOOTH_SKELETON = True
MEDIAN_KERNEL = 7  # Размер медианного фильтра (3, 5, 7, 9)
DILATE_BEFORE_MEDIAN = True  # Расширить скелет перед сглаживанием
DILATE_KERNEL = 5  # Размер расширения (3, 5, 7)

# Удаление изолированных точек
MIN_COMPONENT_SIZE = 10

print("=" * 60)
print("СГЛАЖИВАНИЕ СКЕЛЕТА (Сохранение структуры)")
print("=" * 60)
print(f"Входной файл: {INPUT_IMAGE}")
print(f"Выходной файл: {OUTPUT_SKELETON}")
print(f"Median kernel: {MEDIAN_KERNEL}x{MEDIAN_KERNEL}")
if DILATE_BEFORE_MEDIAN:
    print(f"Dilate перед median: {DILATE_KERNEL}x{DILATE_KERNEL}")

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
# ШАГ 2: Бинаризация и скелетизация
# ====================
print("\n[2/5] Бинаризация и скелетизация...")
_, binary_white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

if alpha_mask is not None:
    binary_white[~alpha_mask] = 0

binary_black = cv2.bitwise_not(binary_white)
if alpha_mask is not None:
    binary_black[~alpha_mask] = 0

# Скелетизация оригинальных границ
skeleton_bool = skeletonize(binary_black > 0)
skeleton = (skeleton_bool * 255).astype(np.uint8)

print(f"Пикселей в исходном скелете: {np.sum(skeleton > 0)}")

cv2.imwrite('debug_01_original_skeleton.png', skeleton)
print("→ debug_01_original_skeleton.png (исходный скелет)")

# ====================
# ШАГ 3: Удаление изолированных точек
# ====================
print(f"\n[3/5] Удаление изолированных точек...")
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
# ШАГ 4: Сглаживание скелета
# ====================
if SMOOTH_SKELETON:
    print(f"\n[4/5] Сглаживание скелета...")

    # Шаг 1: Немного расширить скелет (заполнить мелкие зигзаги)
    if DILATE_BEFORE_MEDIAN:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_KERNEL, DILATE_KERNEL))
        skeleton_dilated = cv2.dilate(skeleton, kernel, iterations=1)

        cv2.imwrite('debug_02_dilated_skeleton.png', skeleton_dilated)
        print(f"  → debug_02_dilated_skeleton.png (расширенный скелет)")
    else:
        skeleton_dilated = skeleton

    # Шаг 2: Медианный фильтр (сглаживание)
    skeleton_median = cv2.medianBlur(skeleton_dilated, MEDIAN_KERNEL)

    cv2.imwrite('debug_03_median.png', skeleton_median)
    print(f"  → debug_03_median.png (после median filter)")

    # Шаг 3: Бинаризация
    _, skeleton_binary = cv2.threshold(skeleton_median, 127, 255, cv2.THRESH_BINARY)

    cv2.imwrite('debug_04_binary.png', skeleton_binary)
    print(f"  → debug_04_binary.png (бинаризация)")

    # Шаг 4: Повторная скелетизация (возврат к 1px)
    skeleton_smooth_bool = skeletonize(skeleton_binary > 0)
    skeleton = (skeleton_smooth_bool * 255).astype(np.uint8)

    print(f"  Пикселей после сглаживания: {np.sum(skeleton > 0)}")

    cv2.imwrite('debug_05_smoothed_skeleton.png', skeleton)
    print(f"  → debug_05_smoothed_skeleton.png (сглаженный скелет 1px)")

else:
    print("\n[4/5] Сглаживание пропущено")

# ====================
# Сохранение
# ====================
print(f"\n[5/5] Сохранение результата...")

output = cv2.bitwise_not(skeleton)
if alpha_mask is not None:
    output[~alpha_mask] = 255

cv2.imwrite(OUTPUT_SKELETON, output)
print(f"→ {OUTPUT_SKELETON}")
print("Формат: черные линии на белом фоне")

print("\n" + "=" * 60)
print("ГОТОВО!")
print("=" * 60)
print(f"\nДля QGIS:")
print(f"  Raster → Conversion → Polygonize")
print(f"  Файл: {OUTPUT_SKELETON}")
print("\nЕсли нужно сильнее сгладить:")
print(f"  Увеличьте MEDIAN_KERNEL до 5 или 7")
print(f"  Увеличьте DILATE_KERNEL до 5")
print("=" * 60)
