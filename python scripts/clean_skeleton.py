import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy import ndimage

# ====================
# ПАРАМЕТРЫ
# ====================
INPUT_IMAGE = 'map_1891_transparent.png'
OUTPUT_SKELETON = 'map_1891_skeleton_clean.png'

# Удаление изолированных точек
REMOVE_ISOLATED = True
MIN_COMPONENT_SIZE = 10  # Минимальный размер компонента

# Удаление коротких "веток" (выступов)
REMOVE_SHORT_BRANCHES = True
BRANCH_ITERATIONS = 3  # Количество итераций pruning (1-5)

# Очень легкое сглаживание
LIGHT_SMOOTHING = False  # НЕ работает на скелете 1px!
SMOOTH_METHOD = 'none'  # 'opening', 'closing', 'none'
SMOOTH_KERNEL_SIZE = 3  # Размер kernel (3 - минимум)

print("=" * 60)
print("ЛЕГКАЯ ОЧИСТКА СКЕЛЕТА")
print("=" * 60)
print(f"Входной файл: {INPUT_IMAGE}")
print(f"Выходной файл: {OUTPUT_SKELETON}")
print(f"Удаление изолированных точек: {REMOVE_ISOLATED}")
print(f"Удаление коротких веток: {REMOVE_SHORT_BRANCHES} ({BRANCH_ITERATIONS} итераций)")
print(f"Легкое сглаживание: {LIGHT_SMOOTHING} ({SMOOTH_METHOD})")

# ====================
# ШАГ 1: Загрузка
# ====================
print("\n[1/4] Загрузка изображения...")
img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(f"Файл {INPUT_IMAGE} не найден!")

height, width = img.shape[:2]
print(f"Размер: {width}x{height} пикселей")

alpha_mask = None
if len(img.shape) == 3 and img.shape[2] == 4:
    alpha = img[:, :, 3]
    rgb = img[:, :, :3]
    alpha_mask = alpha > 50
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
else:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

# Бинаризация и скелетизация
print("\n[2/4] Бинаризация и скелетизация...")
_, binary_white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

if alpha_mask is not None:
    binary_white[~alpha_mask] = 0

binary_black = cv2.bitwise_not(binary_white)
if alpha_mask is not None:
    binary_black[~alpha_mask] = 0

skeleton_bool = skeletonize(binary_black > 0)
skeleton = (skeleton_bool * 255).astype(np.uint8)

original_pixels = np.sum(skeleton > 0)
print(f"Пикселей в исходном скелете: {original_pixels}")

cv2.imwrite('debug_01_original.png', skeleton)
print("→ debug_01_original.png")

# ====================
# ШАГ 2: Удаление изолированных точек
# ====================
if REMOVE_ISOLATED:
    print(f"\n[3/4] Удаление изолированных точек (мин. размер: {MIN_COMPONENT_SIZE}px)...")

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
    pixels_after_cleanup = np.sum(skeleton > 0)

    print(f"  Удалено компонент: {removed_count}")
    print(f"  Пикселей после: {pixels_after_cleanup}")

    cv2.imwrite('debug_02_no_isolated.png', skeleton)
    print("  → debug_02_no_isolated.png")

# ====================
# ШАГ 3: Удаление коротких веток (pruning)
# ====================
if REMOVE_SHORT_BRANCHES:
    print(f"\n[4/4] Удаление коротких веток ({BRANCH_ITERATIONS} итераций)...")

    skeleton_pruned = skeleton.copy()
    kernel_neighbors = np.ones((3, 3), dtype=np.uint8)

    total_removed = 0
    for iteration in range(BRANCH_ITERATIONS):
        skeleton_binary = (skeleton_pruned > 0).astype(np.uint8)

        # Подсчет соседей
        neighbor_count = ndimage.convolve(skeleton_binary, kernel_neighbors, mode='constant', cval=0)

        # Концевые точки: пиксель = 1, соседей = 1 (всего = 2)
        endpoints = (skeleton_binary == 1) & (neighbor_count == 2)

        skeleton_pruned[endpoints] = 0
        removed = np.sum(endpoints)
        total_removed += removed

        if removed == 0:
            break

    skeleton = skeleton_pruned
    pixels_after_pruning = np.sum(skeleton > 0)

    print(f"  Удалено концевых пикселей: {total_removed}")
    print(f"  Пикселей после: {pixels_after_pruning}")

    cv2.imwrite('debug_03_pruned.png', skeleton)
    print("  → debug_03_pruned.png")

# ====================
# ШАГ 4: Очень легкое сглаживание
# ====================
if LIGHT_SMOOTHING and SMOOTH_METHOD != 'none':
    print(f"\nЛегкое сглаживание ({SMOOTH_METHOD}, kernel {SMOOTH_KERNEL_SIZE}x{SMOOTH_KERNEL_SIZE})...")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SMOOTH_KERNEL_SIZE, SMOOTH_KERNEL_SIZE))

    if SMOOTH_METHOD == 'opening':
        # Opening = erosion + dilation (убирает мелкие выступы)
        smoothed = cv2.morphologyEx(skeleton, cv2.MORPH_OPEN, kernel)
    elif SMOOTH_METHOD == 'closing':
        # Closing = dilation + erosion (заполняет мелкие разрывы)
        smoothed = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel)

    # Повторная скелетизация для возврата к 1px
    smoothed_bool = skeletonize(smoothed > 0)
    skeleton = (smoothed_bool * 255).astype(np.uint8)

    pixels_after_smooth = np.sum(skeleton > 0)
    print(f"  Пикселей после: {pixels_after_smooth}")

    cv2.imwrite('debug_04_smoothed.png', skeleton)
    print("  → debug_04_smoothed.png")

# ====================
# Сохранение
# ====================
print(f"\nСохранение результата...")

final_pixels = np.sum(skeleton > 0)
loss_percent = (1 - final_pixels / original_pixels) * 100

print(f"\nИтого:")
print(f"  Исходный скелет: {original_pixels} пикселей")
print(f"  Финальный скелет: {final_pixels} пикселей")
print(f"  Потеря: {loss_percent:.1f}%")

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
print("\nНастройка:")
print(f"  BRANCH_ITERATIONS: больше = меньше выступов")
print(f"  SMOOTH_METHOD: 'opening' (убирает выступы) или 'closing' (заполняет разрывы)")
print("=" * 60)
