import cv2
import numpy as np
from skimage.morphology import skeletonize
import json

# ====================
# ПАРАМЕТРЫ
# ====================
INPUT_IMAGE = 'map_1891_transparent.png'
OUTPUT_SKELETON = 'map_1891_skeleton_straight.png'
OUTPUT_GEOJSON = 'map_1891_lines.geojson'

# Упрощение контуров
SIMPLIFY_EPSILON = 8.0  # Пикселей отклонения (чем больше - тем прямее)

# Удаление изолированных точек
MIN_COMPONENT_SIZE = 10

# Минимальная длина контура
MIN_CONTOUR_LENGTH = 15

print("=" * 60)
print("УПРОЩЕНИЕ КОНТУРОВ И ПЕРЕРИСОВКА ПРЯМЫМИ ЛИНИЯМИ")
print("=" * 60)
print(f"Входной файл: {INPUT_IMAGE}")
print(f"Выходной PNG: {OUTPUT_SKELETON}")
print(f"Выходной GeoJSON: {OUTPUT_GEOJSON}")
print(f"Epsilon упрощения: {SIMPLIFY_EPSILON} пикселей")

# ====================
# ШАГ 1: Загрузка и скелетизация
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

print("\n[2/5] Бинаризация и скелетизация...")
_, binary_white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

if alpha_mask is not None:
    binary_white[~alpha_mask] = 0

binary_black = cv2.bitwise_not(binary_white)
if alpha_mask is not None:
    binary_black[~alpha_mask] = 0

skeleton_bool = skeletonize(binary_black > 0)
skeleton = (skeleton_bool * 255).astype(np.uint8)

print(f"Пикселей в скелете: {np.sum(skeleton > 0)}")

# Удаление изолированных точек
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

cv2.imwrite('debug_skeleton_original.png', skeleton)
print("→ debug_skeleton_original.png")

# ====================
# ШАГ 2: Извлечение контуров
# ====================
print(f"\n[4/5] Извлечение и упрощение контуров...")

contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print(f"Найдено контуров: {len(contours)}")

# Упрощение контуров
simplified_contours = []
geojson_features = []
vertices_before = 0
vertices_after = 0
skipped_short = 0

for i, contour in enumerate(contours):
    # Длина контура
    perimeter = cv2.arcLength(contour, False)

    if perimeter < MIN_CONTOUR_LENGTH:
        skipped_short += 1
        continue

    vertices_before += len(contour)

    # Упростить контур (Douglas-Peucker)
    epsilon = SIMPLIFY_EPSILON
    approx = cv2.approxPolyDP(contour, epsilon, False)

    vertices_after += len(approx)
    simplified_contours.append(approx)

    # Создать LineString для GeoJSON
    coordinates = [[int(point[0][0]), int(point[0][1])] for point in approx]

    feature = {
        "type": "Feature",
        "properties": {
            "id": i,
            "length": float(perimeter),
            "vertices_original": int(len(contour)),
            "vertices_simplified": int(len(approx))
        },
        "geometry": {
            "type": "LineString",
            "coordinates": coordinates
        }
    }

    geojson_features.append(feature)

print(f"\nРезультаты упрощения:")
print(f"  Всего контуров: {len(contours)}")
print(f"  Пропущено коротких: {skipped_short}")
print(f"  Сохранено контуров: {len(simplified_contours)}")
print(f"  Вершин до: {vertices_before}")
print(f"  Вершин после: {vertices_after}")
if vertices_before > 0:
    reduction = (1 - vertices_after / vertices_before) * 100
    print(f"  Сокращение: {reduction:.1f}%")

# ====================
# ШАГ 3: Перерисовка упрощенных контуров
# ====================
print(f"\n[5/5] Перерисовка упрощенных контуров прямыми линиями...")

# Создать белое изображение
output_img = np.ones((height, width), dtype=np.uint8) * 255

# Рисовать каждый упрощенный контур как ломаную линию
for contour in simplified_contours:
    # polylines соединяет точки прямыми линиями
    cv2.polylines(output_img, [contour], isClosed=False, color=0, thickness=1)

# Применить альфа-маску
if alpha_mask is not None:
    output_img[~alpha_mask] = 255

cv2.imwrite(OUTPUT_SKELETON, output_img)
print(f"→ {OUTPUT_SKELETON}")
print("Формат: прямые черные линии на белом фоне")

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

print(f"→ {OUTPUT_GEOJSON} ({len(geojson_features)} контуров)")

print("\n" + "=" * 60)
print("ГОТОВО!")
print("=" * 60)
print(f"\nСоздано:")
print(f"  {OUTPUT_SKELETON} - упрощенные прямые линии")
print(f"  {OUTPUT_GEOJSON} - векторный формат для QGIS")
print(f"\nВсе линии между точками - ИДЕАЛЬНО ПРЯМЫЕ")
print(f"Сокращение вершин: {reduction:.1f}%")
print("\nДля QGIS:")
print(f"  Vector: {OUTPUT_GEOJSON}")
print(f"  Raster: {OUTPUT_SKELETON} → Polygonize")
print("\nЕсли нужно еще прямее:")
print(f"  Увеличьте SIMPLIFY_EPSILON до 10-15")
print("=" * 60)
