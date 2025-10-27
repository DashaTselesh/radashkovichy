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

# Hough Lines параметры
HOUGH_THRESHOLD = 30  # Минимальное количество пикселей для линии
HOUGH_MIN_LINE_LENGTH = 20  # Минимальная длина линии
HOUGH_MAX_LINE_GAP = 5  # Максимальный разрыв для соединения линий

# Удаление изолированных точек
MIN_COMPONENT_SIZE = 10  # Минимальный размер компонента в пикселях

print("=" * 60)
print("СОЗДАНИЕ ПРЯМЫХ ЛИНИЙ (Hough Transform)")
print("=" * 60)
print(f"Входной файл: {INPUT_IMAGE}")
print(f"Выходной GeoJSON: {OUTPUT_GEOJSON}")
print(f"Выходной PNG: {OUTPUT_PNG}")
print(f"Hough threshold: {HOUGH_THRESHOLD}")
print(f"Мин. длина линии: {HOUGH_MIN_LINE_LENGTH}")
print(f"Макс. разрыв: {HOUGH_MAX_LINE_GAP}")

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

cv2.imwrite('debug_skeleton_cleaned.png', skeleton)
print("→ debug_skeleton_cleaned.png")

# ====================
# ШАГ 3: Применение Hough Lines
# ====================
print(f"\n[4/5] Применение Hough Lines для нахождения прямых...")

# Probabilistic Hough Line Transform
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

# ====================
# ШАГ 4: Создание GeoJSON
# ====================
print(f"\n[5/5] Создание GeoJSON и PNG...")

geojson_features = []

for i, line in enumerate(lines):
    x1, y1, x2, y2 = line

    # Вычислить длину линии
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Создать LineString для GeoJSON
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

# ====================
# Сохранение GeoJSON
# ====================
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

# ====================
# Отрисовка прямых линий в PNG
# ====================
print(f"\nОтрисовка прямых линий в PNG...")

# Создать белое изображение
output_img = np.ones((height, width), dtype=np.uint8) * 255

# Отрисовать прямые линии
for line in lines:
    x1, y1, x2, y2 = line
    cv2.line(output_img, (x1, y1), (x2, y2), 0, 1)

# Применить альфа-маску
if alpha_mask is not None:
    output_img[~alpha_mask] = 255

cv2.imwrite(OUTPUT_PNG, output_img)
print(f"→ {OUTPUT_PNG}")
print("Формат: прямые черные линии на белом фоне")

print("\n" + "=" * 60)
print("ГОТОВО!")
print("=" * 60)
print(f"\nНайдено {len(lines)} прямых сегментов")
print("Все линии идеально прямые (2 точки на линию)")
print("\nДальнейшие шаги в QGIS:")
print(f"1. Layer → Add Layer → Add Vector Layer")
print(f"2. Выберите файл: {OUTPUT_GEOJSON}")
print(f"3. Каждая линия = прямой сегмент (2 вершины)")
print("\nЕсли линий слишком много/мало, настройте:")
print(f"  HOUGH_THRESHOLD = {HOUGH_THRESHOLD} (↑ = меньше линий)")
print(f"  HOUGH_MAX_LINE_GAP = {HOUGH_MAX_LINE_GAP} (↑ = длиннее линии)")
print("=" * 60)
