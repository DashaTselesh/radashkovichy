import cv2
import numpy as np
from skimage.morphology import skeletonize
import json

# ====================
# ПАРАМЕТРЫ
# ====================
INPUT_IMAGE = 'map_1891_transparent.png'
OUTPUT_SKELETON = 'map_1891_skeleton_smooth.png'
OUTPUT_GEOJSON = 'map_1891_lines.geojson'
OUTPUT_PNG = 'map_1891_lines_simplified.png'

# Сглаживание скелета
DILATE_KERNEL_SIZE = 9  # Размер расширения (3, 5, 7, 9, 11)
BLUR_KERNEL_SIZE = 11  # Размер размытия (3, 5, 7, 9, 11, 13)
DILATE_ITERATIONS = 2  # Количество итераций дилатации

# Hough Lines параметры
USE_HOUGH = True
HOUGH_THRESHOLD = 15  # Минимальное количество пикселей для линии
HOUGH_MIN_LINE_LENGTH = 10  # Минимальная длина линии
HOUGH_MAX_LINE_GAP = 15  # Максимальный разрыв для соединения линий (выше = меньше пробелов)

# Удаление изолированных точек
MIN_COMPONENT_SIZE = 10  # Минимальный размер компонента в пикселях

print("=" * 60)
print("СОЗДАНИЕ СГЛАЖЕННОГО СКЕЛЕТА И ПРЯМЫХ ЛИНИЙ")
print("=" * 60)
print(f"Входной файл: {INPUT_IMAGE}")
print(f"Выходной скелет: {OUTPUT_SKELETON}")
print(f"Выходной GeoJSON: {OUTPUT_GEOJSON}")
print(f"Выходной PNG: {OUTPUT_PNG}")
print(f"Dilate kernel: {DILATE_KERNEL_SIZE}x{DILATE_KERNEL_SIZE} ({DILATE_ITERATIONS} итераций)")
print(f"Blur kernel: {BLUR_KERNEL_SIZE}x{BLUR_KERNEL_SIZE}")

# ====================
# ШАГ 1: Загрузка
# ====================
print("\n[1/6] Загрузка изображения...")
img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(f"Файл {INPUT_IMAGE} не найден!")

height, width = img.shape[:2]
print(f"Размер: {width}x{height} пикселей")

# Обработка альфа-канала
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

# Бинаризация
print("\n[2/6] Бинаризация...")
_, binary_white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

if alpha_mask is not None:
    binary_white[~alpha_mask] = 0

# Границы (черные линии)
binary_black = cv2.bitwise_not(binary_white)
if alpha_mask is not None:
    binary_black[~alpha_mask] = 0

# ====================
# ШАГ 2: Сглаживание границ ПЕРЕД скелетизацией
# ====================
print(f"\n[3/6] Сглаживание границ (dilate + blur)...")

# Дилатация - расширяем линии (убирает мелкие зигзаги)
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE))
dilated = cv2.dilate(binary_black, kernel_dilate, iterations=DILATE_ITERATIONS)

cv2.imwrite('debug_01_dilated.png', dilated)
print(f"→ debug_01_dilated.png (после дилатации)")

# Размытие - сглаживаем края
blurred = cv2.GaussianBlur(dilated, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)

cv2.imwrite('debug_02_blurred.png', blurred)
print(f"→ debug_02_blurred.png (после размытия)")

# Бинаризация после размытия
_, binary_smooth = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

cv2.imwrite('debug_03_binary_smooth.png', binary_smooth)
print(f"→ debug_03_binary_smooth.png (бинаризация)")

# ====================
# ШАГ 3: Скелетизация сглаженных границ
# ====================
print(f"\n[4/6] Скелетизация сглаженных границ...")

skeleton_bool = skeletonize(binary_smooth > 0)
skeleton = (skeleton_bool * 255).astype(np.uint8)

print(f"Пикселей в скелете: {np.sum(skeleton > 0)}")

# ====================
# ШАГ 4: Удаление изолированных точек
# ====================
print(f"\n[5/6] Удаление изолированных точек (мин. размер: {MIN_COMPONENT_SIZE}px)...")

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

# Сохранить сглаженный скелет
output_skeleton = cv2.bitwise_not(skeleton)
if alpha_mask is not None:
    output_skeleton[~alpha_mask] = 255

cv2.imwrite(OUTPUT_SKELETON, output_skeleton)
print(f"→ {OUTPUT_SKELETON} (сглаженный скелет)")

# ====================
# ШАГ 5: Hough Lines на сглаженном скелете
# ====================
if USE_HOUGH:
    print(f"\n[6/6] Применение Hough Lines на сглаженном скелете...")

    lines = cv2.HoughLinesP(
        skeleton,
        rho=1,
        theta=np.pi/180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )

    if lines is None:
        print("⚠️ Линии не найдены! Попробуйте уменьшить HOUGH_THRESHOLD")
        lines = []
    else:
        lines = lines.reshape(-1, 4)
        print(f"Найдено прямых линий: {len(lines)}")

    # Создание GeoJSON
    geojson_features = []

    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        coordinates = [[int(x1), int(y1)], [int(x2), int(y2)]]

        feature = {
            "type": "Feature",
            "properties": {
                "id": i,
                "length": float(length)
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates
            }
        }

        geojson_features.append(feature)

    # Сохранение GeoJSON
    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
            }
        },
        "features": geojson_features
    }

    with open(OUTPUT_GEOJSON, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)

    print(f"→ {OUTPUT_GEOJSON} ({len(geojson_features)} прямых линий)")

    # Отрисовка прямых линий в PNG
    output_img = np.ones((height, width), dtype=np.uint8) * 255

    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(output_img, (x1, y1), (x2, y2), 0, 1)

    if alpha_mask is not None:
        output_img[~alpha_mask] = 255

    cv2.imwrite(OUTPUT_PNG, output_img)
    print(f"→ {OUTPUT_PNG} (прямые линии)")

print("\n" + "=" * 60)
print("ГОТОВО!")
print("=" * 60)
print(f"\nСоздано файлов:")
print(f"  1. {OUTPUT_SKELETON} - сглаженный скелет (БЕЗ пиков)")
if USE_HOUGH:
    print(f"  2. {OUTPUT_GEOJSON} - прямые линии ({len(lines)} сегментов)")
    print(f"  3. {OUTPUT_PNG} - визуализация прямых линий")
print("\nДля QGIS:")
print(f"  Используйте {OUTPUT_GEOJSON} или {OUTPUT_SKELETON}")
print("=" * 60)
