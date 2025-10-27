import cv2
import numpy as np
from skimage.morphology import skeletonize
import json

# ====================
# ПАРАМЕТРЫ
# ====================
INPUT_IMAGE = 'map_1891_transparent.png'
OUTPUT_PNG = 'map_1891_skeleton_corners.png'
OUTPUT_GEOJSON = 'map_1891_lines.geojson'

# Corner detection
CORNER_QUALITY = 0.05  # Качество углов (0.01-0.1, меньше = больше углов)
CORNER_MIN_DISTANCE = 15  # Минимальное расстояние между углами (пикселей)

# Удаление изолированных точек
MIN_COMPONENT_SIZE = 10

print("=" * 60)
print("СОЗДАНИЕ ПРЯМЫХ ЛИНИЙ ИЗ УГЛОВ")
print("=" * 60)
print(f"Входной файл: {INPUT_IMAGE}")
print(f"Выходной PNG: {OUTPUT_PNG}")
print(f"Выходной GeoJSON: {OUTPUT_GEOJSON}")
print(f"Corner quality: {CORNER_QUALITY}")
print(f"Corner min distance: {CORNER_MIN_DISTANCE}")

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
print(f"Удалено компонент: {removed_count}")

cv2.imwrite('debug_skeleton.png', skeleton)

# ====================
# ШАГ 2: Поиск углов
# ====================
print(f"\n[4/5] Поиск углов...")

# Harris Corner Detection на скелете
skeleton_float = np.float32(skeleton)
corners = cv2.goodFeaturesToTrack(
    skeleton_float,
    maxCorners=10000,
    qualityLevel=CORNER_QUALITY,
    minDistance=CORNER_MIN_DISTANCE,
    blockSize=3
)

if corners is None:
    print("⚠️ Углы не найдены!")
    corners = np.array([])
else:
    corners = corners.reshape(-1, 2)
    print(f"Найдено углов: {len(corners)}")

# Визуализация углов
debug_corners = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
for corner in corners:
    x, y = corner.astype(int)
    cv2.circle(debug_corners, (x, y), 2, (0, 0, 255), -1)

cv2.imwrite('debug_corners.png', debug_corners)
print("→ debug_corners.png (найденные углы красным)")

# ====================
# ШАГ 3: Соединение углов в контуры
# ====================
print(f"\n[5/5] Соединение углов в линии...")

# Найти контуры для определения какие углы принадлежат одной линии
contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

print(f"Найдено контуров: {len(contours)}")

geojson_features = []
output_img = np.ones((height, width), dtype=np.uint8) * 255
line_count = 0

for contour_idx, contour in enumerate(contours):
    contour_points = contour.reshape(-1, 2)

    # Найти углы которые принадлежат этому контуру
    # Используем маску контура
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, 1)

    # Расширяем маску для захвата близких углов
    kernel = np.ones((5, 5), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)

    # Найти углы в этой маске
    contour_corners = []
    for corner in corners:
        x, y = corner.astype(int)
        if 0 <= y < height and 0 <= x < width:
            if mask_dilated[y, x] > 0:
                contour_corners.append(corner)

    if len(contour_corners) < 2:
        # Если углов мало, используем концы контура
        contour_corners = [contour_points[0], contour_points[-1]]

    # Сортируем углы вдоль контура
    # Для этого находим ближайшие точки контура для каждого угла
    corner_indices = []
    for corner in contour_corners:
        distances = np.linalg.norm(contour_points - corner, axis=1)
        closest_idx = np.argmin(distances)
        corner_indices.append(closest_idx)

    # Сортируем по индексу на контуре
    sorted_pairs = sorted(zip(corner_indices, contour_corners))
    sorted_corners = [corner for _, corner in sorted_pairs]

    # Соединяем углы прямыми линиями
    for i in range(len(sorted_corners) - 1):
        x1, y1 = sorted_corners[i].astype(int)
        x2, y2 = sorted_corners[i + 1].astype(int)

        # Рисуем линию
        cv2.line(output_img, (x1, y1), (x2, y2), 0, 1)

        # Добавляем в GeoJSON
        feature = {
            "type": "Feature",
            "properties": {
                "id": line_count,
                "contour": contour_idx
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[int(x1), int(y1)], [int(x2), int(y2)]]
            }
        }
        geojson_features.append(feature)
        line_count += 1

print(f"Создано прямых линий: {line_count}")

# ====================
# Сохранение
# ====================
if alpha_mask is not None:
    output_img[~alpha_mask] = 255

cv2.imwrite(OUTPUT_PNG, output_img)
print(f"→ {OUTPUT_PNG}")

# GeoJSON
geojson = {
    "type": "FeatureCollection",
    "features": geojson_features
}

with open(OUTPUT_GEOJSON, 'w', encoding='utf-8') as f:
    json.dump(geojson, f, indent=2)

print(f"→ {OUTPUT_GEOJSON} ({len(geojson_features)} линий)")

print("\n" + "=" * 60)
print("ГОТОВО!")
print("=" * 60)
print(f"\nСоздано {line_count} прямых сегментов из {len(corners)} углов")
print("\nДля QGIS:")
print(f"  Vector: {OUTPUT_GEOJSON}")
print(f"  Raster: {OUTPUT_PNG}")
print("\nЕсли нужно больше/меньше углов:")
print(f"  CORNER_QUALITY: ↓ = больше углов, ↑ = меньше углов")
print("=" * 60)
