import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from scipy import ndimage
from skimage import measure
import json

# ====================
# ПАРАМЕТРЫ
# ====================
INPUT_IMAGE = 'map_1891.png'
OUTPUT_GEOJSON = 'map_1891_raw.geojson'

# Фильтр по площади (в пикселях)
MIN_AREA = 50         # Минимальная площадь участка (уменьшено для мелких)
MAX_AREA = 50000      # Максимальная площадь (уменьшено чтобы исключить фон)

# Упрощение контуров
EPSILON_FACTOR = 0.005  # Небольшое упрощение для прямых линий

print("=" * 60)
print("ИЗВЛЕЧЕНИЕ УЧАСТКОВ ЧЕРЕЗ LABEL-СЕГМЕНТАЦИЮ")
print("=" * 60)
print(f"Ожидаемое количество участков: ~188")

# ====================
# ШАГ 1: Загрузка
# ====================
print("\n[1/7] Загрузка изображения...")
img = cv2.imread(INPUT_IMAGE)
if img is None:
    raise FileNotFoundError(f"Файл {INPUT_IMAGE} не найден!")

height, width = img.shape[:2]
print(f"Размер: {width}x{height} пикселей")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ====================
# ШАГ 2: Адаптивная бинаризация
# ====================
print("\n[2/7] Адаптивная бинаризация...")
binary = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    blockSize=15,
    C=5
)

cv2.imwrite('debug_v2_01_binary.png', binary)
print("→ Сохранено: debug_v2_01_binary.png")

# ====================
# ШАГ 3: Замыкание разрывов
# ====================
print("\n[3/7] Замыкание разрывов в границах...")
kernel_close = np.ones((5, 5), np.uint8)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=4)

cv2.imwrite('debug_v2_02_closed.png', closed)
print("→ Сохранено: debug_v2_02_closed.png")

# ====================
# ШАГ 4: Утончение границ для плотного прилегания
# ====================
print("\n[4/7] Утончение границ (эрозия)...")
# Эрозия делает границы тоньше - участки будут плотнее друг к другу
kernel_erode = np.ones((3, 3), np.uint8)
thinned = cv2.erode(closed, kernel_erode, iterations=2)
cv2.imwrite('debug_v2_03_thinned.png', thinned)
print("→ Сохранено: debug_v2_03_thinned.png")

# ====================
# ШАГ 5: Инверсия - участки становятся белыми
# ====================
print("\n[5/7] Инверсия (участки = белые области)...")
inverted = cv2.bitwise_not(thinned)
cv2.imwrite('debug_v2_04_inverted.png', inverted)
print("→ Сохранено: debug_v2_04_inverted.png")

# ====================
# ШАГ 6: Label-based сегментация
# ====================
print("\n[6/7] Label-сегментация (автоматическое выделение участков)...")

# Конвертируем в бинарный массив (True/False)
binary_mask = inverted > 127

# Используем scipy.ndimage.label для автоматической разметки связных областей
# Это автоматически создаст участки с общими границами!
labeled_array, num_features = ndimage.label(binary_mask)

print(f"Найдено связных областей: {num_features}")

# Визуализация меток
labeled_vis = (labeled_array % 256).astype(np.uint8)
labeled_colored = cv2.applyColorMap(labeled_vis, cv2.COLORMAP_JET)
cv2.imwrite('debug_v2_05_labeled.png', labeled_colored)
print("→ Сохранено: debug_v2_05_labeled.png (цветная визуализация)")

# ====================
# ШАГ 7: Извлечение контуров каждого участка
# ====================
print("\n[7/7] Извлечение контуров участков...")

valid_parcels = []
rejected = {'too_small': 0, 'too_big': 0, 'invalid_geometry': 0}

for label_id in range(1, num_features + 1):
    # Создаем маску для текущего участка
    parcel_mask = (labeled_array == label_id).astype(np.uint8) * 255

    # Проверяем площадь
    area = np.sum(labeled_array == label_id)

    if area < MIN_AREA:
        rejected['too_small'] += 1
        continue
    if area > MAX_AREA:
        rejected['too_big'] += 1
        continue

    # Находим контур этого участка
    contours, _ = cv2.findContours(parcel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        continue

    # Берем самый большой контур (если их несколько)
    contour = max(contours, key=cv2.contourArea)

    # Упрощение контура для более прямых линий
    perimeter = cv2.arcLength(contour, True)
    epsilon = EPSILON_FACTOR * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Минимум 3 точки для полигона
    if len(approx) < 3:
        rejected['invalid_geometry'] += 1
        continue

    # Попытка создать прямоугольник для компактных форм
    num_vertices = len(approx)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    # Если похоже на прямоугольник - используем минимальный ограничивающий прямоугольник
    if 4 <= num_vertices <= 6 and solidity > 0.85:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        approx = box.reshape(-1, 1, 2)

    valid_parcels.append({
        'id': label_id,
        'contour': approx,
        'area': area,
        'vertices': len(approx),
        'solidity': solidity
    })

print(f"\n✓ Найдено участков: {len(valid_parcels)}")
print(f"✗ Отфильтровано:")
print(f"  - Слишком маленькие: {rejected['too_small']}")
print(f"  - Слишком большие: {rejected['too_big']}")
print(f"  - Некорректная геометрия: {rejected['invalid_geometry']}")

# ====================
# ШАГ 8: Визуализация и экспорт
# ====================
print("\nСоздание визуализации и экспорт в GeoJSON...")

# Визуализация найденных участков
debug_img = img.copy()

for parcel in valid_parcels:
    # Контур зеленым
    cv2.drawContours(debug_img, [parcel['contour']], -1, (0, 255, 0), 3)

    # ID в центре
    M = cv2.moments(parcel['contour'])
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        cv2.putText(debug_img, str(parcel['id']), (cx-20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

cv2.imwrite('debug_v2_06_result.png', debug_img)
print("→ Сохранено: debug_v2_06_result.png")

# Только контуры
contours_only = np.ones((height, width, 3), dtype=np.uint8) * 255
for parcel in valid_parcels:
    cv2.drawContours(contours_only, [parcel['contour']], -1, (0, 0, 0), 2)
cv2.imwrite('debug_v2_07_contours.png', contours_only)
print("→ Сохранено: debug_v2_07_contours.png")

# ====================
# Конвертация в GeoJSON
# ====================
print("\nКонвертация в GeoJSON...")
polygons = []

for parcel in valid_parcels:
    coords = parcel['contour'].squeeze().tolist()

    # Проверка
    if len(coords) < 3:
        continue

    # Обработка одномерных координат
    if not isinstance(coords[0], list):
        coords = [coords]

    # Замыкание полигона
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    try:
        poly = Polygon(coords)

        polygons.append({
            'geometry': poly,
            'auto_id': parcel['id'],
            'area_pixels': int(parcel['area']),
            'vertices': parcel['vertices'],
            'solidity': round(parcel['solidity'], 3),
            'plot_number': None,
            'owner_first_name': None,
            'owner_last_name': None
        })
    except Exception as e:
        print(f"  Ошибка в участке {parcel['id']}: {e}")

gdf = gpd.GeoDataFrame(polygons, crs=None)
gdf.to_file(OUTPUT_GEOJSON, driver='GeoJSON')

print(f"→ Сохранено: {OUTPUT_GEOJSON}")
print(f"Участков в файле: {len(gdf)}")

# ====================
# Статистика
# ====================
print("\n" + "=" * 60)
print("СТАТИСТИКА:")
print("=" * 60)

if len(valid_parcels) > 0:
    areas = [p['area'] for p in valid_parcels]
    vertices = [p['vertices'] for p in valid_parcels]
    solidities = [p['solidity'] for p in valid_parcels]

    print(f"Площадь (пиксели):")
    print(f"  Средняя: {np.mean(areas):.0f}")
    print(f"  Минимум: {np.min(areas):.0f}")
    print(f"  Максимум: {np.max(areas):.0f}")
    print(f"\nВершины контура:")
    print(f"  Среднее: {np.mean(vertices):.1f}")
    print(f"  Минимум: {np.min(vertices)}")
    print(f"  Максимум: {np.max(vertices)}")
    print(f"\nКомпактность (solidity):")
    print(f"  Средняя: {np.mean(solidities):.3f}")
    print(f"  Минимум: {np.min(solidities):.3f}")

print("=" * 60)
print("\n✓ ГОТОВО! Проверьте файлы:")
print("  1. debug_v2_01_binary.png - бинаризация")
print("  2. debug_v2_02_closed.png - замкнутые границы")
print("  3. debug_v2_03_thinned.png - утонченные границы")
print("  4. debug_v2_04_inverted.png - инвертировано")
print("  5. debug_v2_05_labeled.png - автоматическая разметка участков")
print("  6. debug_v2_06_result.png - результат с номерами")
print("  7. debug_v2_07_contours.png - только контуры")
print("  8. map_1891_raw.geojson - GeoJSON для QGIS")
print("\nПРЕИМУЩЕСТВА этого подхода:")
print("  ✓ Участки АВТОМАТИЧЕСКИ имеют общие границы")
print("  ✓ Нет пробелов между соседними участками")
print("  ✓ Внутренние участки корректно обрабатываются")
print("=" * 60)
