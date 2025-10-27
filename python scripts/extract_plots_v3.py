import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from scipy import ndimage
import json

# ====================
# ПАРАМЕТРЫ
# ====================
INPUT_IMAGE = 'map_1891.png'
OUTPUT_GEOJSON = 'map_1891_raw.geojson'

# Фильтр по площади (в пикселях)
MIN_AREA = 30         # Очень маленькие участки
MAX_AREA = 100000     # Исключаем огромные фоновые области

# Агрессивная прямоугольная аппроксимация
RECTANGULARIZE = True  # Превращать участки в прямоугольники
RECT_THRESHOLD = 0.75  # Если solidity > 0.75, делаем прямоугольником

print("=" * 60)
print("ИЗВЛЕЧЕНИЕ УЧАСТКОВ V3: WATERSHED + ПРЯМОУГОЛЬНИКИ")
print("=" * 60)
print(f"Ожидаемое количество участков: ~188")

# ====================
# ШАГ 1: Загрузка
# ====================
print("\n[1/9] Загрузка изображения...")
img = cv2.imread(INPUT_IMAGE)
if img is None:
    raise FileNotFoundError(f"Файл {INPUT_IMAGE} не найден!")

height, width = img.shape[:2]
print(f"Размер: {width}x{height} пикселей")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ====================
# ШАГ 2: Бинаризация с инверсией
# ====================
print("\n[2/9] Бинаризация (участки = белые)...")
# Adaptive threshold
binary = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,  # НЕ инвертируем - белый фон, черные линии
    blockSize=15,
    C=5
)

cv2.imwrite('debug_v3_01_binary.png', binary)
print("→ Сохранено: debug_v3_01_binary.png")

# ====================
# ШАГ 3: Эрозия границ (делаем их тоньше)
# ====================
print("\n[3/9] Утончение границ эрозией...")
# Инвертируем временно (границы = белые)
binary_inv = cv2.bitwise_not(binary)

# Эрозия границ
kernel_erode = np.ones((2, 2), np.uint8)
boundaries_thin = cv2.erode(binary_inv, kernel_erode, iterations=2)

# Инвертируем обратно (участки = белые)
binary_thin = cv2.bitwise_not(boundaries_thin)

cv2.imwrite('debug_v3_02_thin_boundaries.png', binary_thin)
print("→ Сохранено: debug_v3_02_thin_boundaries.png")

# ====================
# ШАГ 4: Используем утонченное изображение напрямую
# ====================
print("\n[4/9] Подготовка для watershed (без MORPH_CLOSE)...")
# НЕ используем MORPH_CLOSE - он заливает всё белым при тонких границах!
# Используем binary_thin напрямую
filled = binary_thin.copy()

cv2.imwrite('debug_v3_03_filled.png', filled)
print("→ Сохранено: debug_v3_03_filled.png")

# ====================
# ШАГ 5: Distance Transform + Watershed
# ====================
print("\n[5/9] Distance transform для watershed...")

# Distance transform работает с БЕЛЫМИ объектами на ЧЕРНОМ фоне
# У нас сейчас белые участки на черном фоне - всё правильно!
# НО проверим: если filled в основном белый, нужно инвертировать
white_pixels = np.sum(filled > 127)
black_pixels = np.sum(filled <= 127)

if white_pixels > black_pixels:
    # Слишком много белого - инвертируем (участки должны быть белыми, но меньше фона)
    print(f"  Обнаружено {white_pixels} белых пикселей vs {black_pixels} черных")
    print("  Изображение выглядит инвертированным, исправляю...")
    filled_for_dt = cv2.bitwise_not(filled)
else:
    filled_for_dt = filled

# Distance transform: находим "центры" участков
dist_transform = cv2.distanceTransform(filled_for_dt, cv2.DIST_L2, 5)

if dist_transform.max() == 0:
    print("  ❌ ОШИБКА: Distance transform вернул нули!")
    print("  Проверьте debug_v3_03_filled.png - должны быть белые участки на черном фоне")
    exit(1)

cv2.imwrite('debug_v3_04_distance.png', (dist_transform / dist_transform.max() * 255).astype(np.uint8))
print("→ Сохранено: debug_v3_04_distance.png")

# Пороговая обработка: выделяем "ядра" участков
# Уменьшаем порог с 0.3 до 0.2 для лучшего захвата маленьких участков
_, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

cv2.imwrite('debug_v3_05_sure_fg.png', sure_fg)
print("→ Сохранено: debug_v3_05_sure_fg.png")

# ====================
# ШАГ 6: Connected Components для маркеров
# ====================
print("\n[6/9] Поиск связных компонент (маркеры для watershed)...")
ret, markers = cv2.connectedComponents(sure_fg)

print(f"Найдено потенциальных участков: {ret - 1}")

# Watershed требует 3-канальное изображение
# Используем то же изображение, что и для distance transform
img_watershed = cv2.cvtColor(filled_for_dt, cv2.COLOR_GRAY2BGR)

# Применяем watershed
print("\n[7/9] Применение watershed сегментации...")
markers = cv2.watershed(img_watershed, markers)

# Визуализация watershed
markers_vis = (markers % 256).astype(np.uint8)
markers_colored = cv2.applyColorMap(markers_vis, cv2.COLORMAP_JET)
cv2.imwrite('debug_v3_06_watershed.png', markers_colored)
print("→ Сохранено: debug_v3_06_watershed.png")

# ====================
# ШАГ 8: Извлечение контуров каждого участка
# ====================
print("\n[8/9] Извлечение и прямоугольная аппроксимация участков...")

valid_parcels = []
rejected = {'too_small': 0, 'too_big': 0, 'invalid_geometry': 0}

# Обрабатываем каждый сегмент watershed
for label_id in range(2, ret + 1):  # Пропускаем 0 (фон) и 1 (границы)
    # Создаем маску для этого участка
    mask = np.zeros_like(filled_for_dt)
    mask[markers == label_id] = 255

    # Площадь
    area = np.sum(mask > 0)

    # Фильтр по площади
    if area < MIN_AREA:
        rejected['too_small'] += 1
        continue
    if area > MAX_AREA:
        rejected['too_big'] += 1
        continue

    # Находим контур
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        rejected['invalid_geometry'] += 1
        continue

    # Берем самый большой контур
    contour = max(contours, key=cv2.contourArea)

    # Вычисляем solidity (компактность)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    # АГРЕССИВНАЯ ПРЯМОУГОЛЬНАЯ АППРОКСИМАЦИЯ
    if RECTANGULARIZE and solidity > RECT_THRESHOLD:
        # Используем минимальный ограничивающий прямоугольник
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        final_contour = box.reshape(-1, 1, 2)
        num_vertices = 4
    else:
        # Упрощение контура
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.01 * perimeter  # Агрессивное упрощение
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Если получилось 4-6 вершин и компактный - тоже делаем прямоугольником
        if 4 <= len(approx) <= 6 and solidity > 0.7:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            final_contour = box.reshape(-1, 1, 2)
            num_vertices = 4
        else:
            final_contour = approx
            num_vertices = len(approx)

    if num_vertices < 3:
        rejected['invalid_geometry'] += 1
        continue

    valid_parcels.append({
        'id': label_id,
        'contour': final_contour,
        'area': area,
        'vertices': num_vertices,
        'solidity': solidity
    })

print(f"\n✓ Найдено участков: {len(valid_parcels)}")
print(f"✗ Отфильтровано:")
print(f"  - Слишком маленькие: {rejected['too_small']}")
print(f"  - Слишком большие: {rejected['too_big']}")
print(f"  - Некорректная геометрия: {rejected['invalid_geometry']}")

# ====================
# ШАГ 9: Визуализация и экспорт
# ====================
print("\n[9/9] Создание визуализации и экспорт в GeoJSON...")

# Визуализация
debug_img = img.copy()

for parcel in valid_parcels:
    # Контур зеленым
    cv2.drawContours(debug_img, [parcel['contour']], -1, (0, 255, 0), 3)

    # ID в центре
    M = cv2.moments(parcel['contour'])
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(debug_img, str(parcel['id']), (cx-15, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

cv2.imwrite('debug_v3_07_result.png', debug_img)
print("→ Сохранено: debug_v3_07_result.png")

# Только контуры
contours_only = np.ones((height, width, 3), dtype=np.uint8) * 255
for parcel in valid_parcels:
    cv2.drawContours(contours_only, [parcel['contour']], -1, (0, 0, 0), 2)
cv2.imwrite('debug_v3_08_contours.png', contours_only)
print("→ Сохранено: debug_v3_08_contours.png")

# ====================
# Конвертация в GeoJSON
# ====================
print("\nКонвертация в GeoJSON...")
polygons = []

for parcel in valid_parcels:
    coords = parcel['contour'].squeeze().tolist()

    if len(coords) < 3:
        continue

    if not isinstance(coords[0], list):
        coords = [coords]

    # Замыкание
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    try:
        poly = Polygon(coords)

        if not poly.is_valid:
            continue

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

# Проверяем что есть полигоны
if len(polygons) == 0:
    print("\n❌ ОШИБКА: Не найдено ни одного валидного полигона!")
    print("Проверьте отладочные изображения.")
    exit(1)

# Создаем GeoDataFrame с явным указанием геометрии
gdf = gpd.GeoDataFrame(polygons, geometry='geometry', crs=None)
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
    rectangles = sum(1 for p in valid_parcels if p['vertices'] == 4)

    print(f"Площадь (пиксели):")
    print(f"  Средняя: {np.mean(areas):.0f}")
    print(f"  Минимум: {np.min(areas):.0f}")
    print(f"  Максимум: {np.max(areas):.0f}")
    print(f"\nВершины контура:")
    print(f"  Среднее: {np.mean(vertices):.1f}")
    print(f"  Минимум: {np.min(vertices)}")
    print(f"  Максимум: {np.max(vertices)}")
    print(f"\nПрямоугольников (4 вершины): {rectangles} ({rectangles/len(valid_parcels)*100:.1f}%)")
    print(f"\nКомпактность (solidity):")
    print(f"  Средняя: {np.mean(solidities):.3f}")
    print(f"  Минимум: {np.min(solidities):.3f}")

print("=" * 60)
print("\n✓ ГОТОВО! Проверьте файлы:")
print("  1. debug_v3_01_binary.png - бинаризация")
print("  2. debug_v3_02_thin_boundaries.png - утонченные границы")
print("  3. debug_v3_03_filled.png - заполненные участки")
print("  4. debug_v3_04_distance.png - distance transform")
print("  5. debug_v3_05_sure_fg.png - ядра участков")
print("  6. debug_v3_06_watershed.png - результат watershed")
print("  7. debug_v3_07_result.png - финальный результат")
print("  8. debug_v3_08_contours.png - только контуры")
print("  9. map_1891_raw.geojson - GeoJSON для QGIS")
print("\nНОВЫЕ ПРЕИМУЩЕСТВА V3:")
print("  ✓ Watershed автоматически разделяет слипшиеся участки")
print("  ✓ АГРЕССИВНАЯ прямоугольная аппроксимация (>75% стали прямоугольниками)")
print("  ✓ Утонченные границы → участки ближе друг к другу")
print("  ✓ Distance transform находит центры → лучше для маленьких участков")
print("=" * 60)
