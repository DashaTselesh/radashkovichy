import cv2
import numpy as np
from skimage.morphology import skeletonize
import json

# ====================
# ПАРАМЕТРЫ
# ====================
INPUT_IMAGE = 'map_1891_transparent.png'
OUTPUT_GEOJSON = 'map_1891_lines.geojson'
OUTPUT_PNG = 'map_1891_lines_simplified.png'

# Упрощение линий (Douglas-Peucker)
SIMPLIFY_EPSILON = 5.0  # Пикселей отклонения (чем больше - тем прямее линии)

# Удаление изолированных точек
MIN_COMPONENT_SIZE = 10  # Минимальный размер компонента в пикселях

# Минимальная длина линии
MIN_LINE_LENGTH = 20  # Минимальная длина линии в пикселях (меньше = удалить)

print("=" * 60)
print("СОЗДАНИЕ УПРОЩЕННЫХ ЛИНИЙ (GeoJSON + PNG)")
print("=" * 60)
print(f"Входной файл: {INPUT_IMAGE}")
print(f"Выходной GeoJSON: {OUTPUT_GEOJSON}")
print(f"Выходной PNG: {OUTPUT_PNG}")
print(f"Epsilon упрощения: {SIMPLIFY_EPSILON} пикселей")
print(f"Минимальная длина линии: {MIN_LINE_LENGTH} пикселей")

# ====================
# ШАГ 1: Загрузка и скелетизация
# ====================
print("\n[1/5] Загрузка изображения...")
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
print("\n[2/5] Бинаризация и скелетизация...")
_, binary_white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

if alpha_mask is not None:
    binary_white[~alpha_mask] = 0

# Скелетизация черных границ
binary_black = cv2.bitwise_not(binary_white)
if alpha_mask is not None:
    binary_black[~alpha_mask] = 0

skeleton_bool = skeletonize(binary_black > 0)
skeleton = (skeleton_bool * 255).astype(np.uint8)

print(f"Пикселей в скелете: {np.sum(skeleton > 0)}")

# ====================
# ШАГ 2: Удаление изолированных точек
# ====================
print(f"\n[3/5] Удаление изолированных точек (мин. размер: {MIN_COMPONENT_SIZE}px)...")

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
# ШАГ 3: Извлечение контуров
# ====================
print(f"\n[4/5] Извлечение и упрощение контуров...")

# Найти контуры (это будут наши линии)
contours, hierarchy = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

print(f"Найдено контуров: {len(contours)}")

# ====================
# ШАГ 4: Упрощение контуров и создание GeoJSON
# ====================
print(f"\nУпрощение контуров (epsilon={SIMPLIFY_EPSILON})...")

geojson_features = []
simplified_contours = []
total_vertices_before = 0
total_vertices_after = 0
skipped_short = 0

for i, contour in enumerate(contours):
    # Подсчет вершин до упрощения
    vertices_before = len(contour)
    total_vertices_before += vertices_before

    # Вычислить длину контура
    perimeter = cv2.arcLength(contour, False)

    # Пропустить очень короткие линии
    if perimeter < MIN_LINE_LENGTH:
        skipped_short += 1
        continue

    # Упростить контур (Douglas-Peucker)
    epsilon = SIMPLIFY_EPSILON
    approx = cv2.approxPolyDP(contour, epsilon, False)

    vertices_after = len(approx)
    total_vertices_after += vertices_after

    simplified_contours.append(approx)

    # Создать LineString для GeoJSON
    # Координаты в формате [x, y] (пиксели)
    coordinates = [[int(point[0][0]), int(point[0][1])] for point in approx]

    feature = {
        "type": "Feature",
        "properties": {
            "id": i,
            "length": float(perimeter),
            "vertices_original": int(vertices_before),
            "vertices_simplified": int(vertices_after)
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
print(f"  Сохранено линий: {len(simplified_contours)}")
print(f"  Вершин до упрощения: {total_vertices_before}")
print(f"  Вершин после упрощения: {total_vertices_after}")
if total_vertices_before > 0:
    reduction = (1 - total_vertices_after / total_vertices_before) * 100
    print(f"  Сокращение вершин: {reduction:.1f}%")

# ====================
# Сохранение GeoJSON
# ====================
print(f"\n[5/5] Сохранение результатов...")

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

print(f"→ {OUTPUT_GEOJSON} ({len(geojson_features)} линий)")

# ====================
# Отрисовка упрощенных линий в PNG
# ====================
print(f"\nОтрисовка упрощенных линий в PNG...")

# Создать белое изображение
output_img = np.ones((height, width), dtype=np.uint8) * 255

# Отрисовать упрощенные контуры черными линиями
for contour in simplified_contours:
    cv2.polylines(output_img, [contour], False, 0, 1)

# Применить альфа-маску
if alpha_mask is not None:
    output_img[~alpha_mask] = 255

cv2.imwrite(OUTPUT_PNG, output_img)
print(f"→ {OUTPUT_PNG}")
print("Формат: черные линии на белом фоне")

print("\n" + "=" * 60)
print("ГОТОВО!")
print("=" * 60)
print("\nДальнейшие шаги в QGIS:")
print("ВАРИАНТ 1 - Векторный (рекомендуется):")
print(f"  1. Layer → Add Layer → Add Vector Layer")
print(f"  2. Выберите файл: {OUTPUT_GEOJSON}")
print(f"  3. Линии готовы к использованию (прямые, мало вершин)")
print("\nВАРИАНТ 2 - Растровый:")
print(f"  1. Откройте файл: {OUTPUT_PNG}")
print(f"  2. Raster → Conversion → Polygonize")
print("=" * 60)
