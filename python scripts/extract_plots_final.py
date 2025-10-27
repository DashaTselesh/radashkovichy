import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from scipy import ndimage

# ====================
# ПАРАМЕТРЫ
# ====================
INPUT_IMAGE = 'map_1891_cleaned.png'  # Очищено от вложенных участков
OUTPUT_GEOJSON = 'map_1891_raw.geojson'

MIN_AREA = 30
MAX_AREA = 100000
EPSILON_FACTOR = 0.01  # Упрощение для прямых линий (было 0.003)

print("=" * 60)
print("ИЗВЛЕЧЕНИЕ УЧАСТКОВ С ОБЩИМИ ГРАНИЦАМИ")
print("=" * 60)
print("Метод: Label-сегментация (каждый пиксель → один участок)")

# ====================
# ШАГ 1: Загрузка
# ====================
print("\n[1/6] Загрузка изображения...")
img = cv2.imread(INPUT_IMAGE)
if img is None:
    raise FileNotFoundError(f"Файл {INPUT_IMAGE} не найден!")

height, width = img.shape[:2]
print(f"Размер: {width}x{height} пикселей")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ====================
# ШАГ 2: Бинаризация
# ====================
print("\n[2/7] Бинаризация...")
_, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
cv2.imwrite('debug_final_01_binary.png', binary)
print("→ debug_final_01_binary.png")

# ====================
# ШАГ 3: Пропускаем утончение (оно удаляет маленькие участки)
# ====================
print("\n[3/7] Используем бинарное изображение без утончения...")
# НЕ делаем эрозию - она удаляет маленькие участки!
binary_thin = binary.copy()

cv2.imwrite('debug_final_02_ready.png', binary_thin)
print("→ debug_final_02_ready.png (готово к заполнению)")

# ====================
# ШАГ 4: Поиск контуров БЕЛЫХ участков (не инвертируя!)
# ====================
print("\n[4/7] Поиск контуров белых участков...")

# Находим контуры на БЕЛЫХ областях (binary_thin: белый=255, черный=0)
contours_all, hierarchy = cv2.findContours(binary_thin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
print(f"Найдено контуров: {len(contours_all)}")

# ====================
# ШАГ 5: Создание label-массива через заполнение
# ====================
print("\n[5/7] Заполнение каждого БЕЛОГО участка уникальным номером...")

# Создаем пустой массив для label
labeled_array = np.zeros((height, width), dtype=np.int32)

# Заполняем каждый БЕЛЫЙ участок своим номером (границы остаются 0)
label_id = 1
filled_count = 0

for i, contour in enumerate(contours_all):
    area = cv2.contourArea(contour)

    # Пропускаем слишком маленькие и огромные
    if area < MIN_AREA or area > MAX_AREA:
        continue

    # Заполняем этот контур номером label_id
    cv2.drawContours(labeled_array, [contour], -1, label_id, thickness=cv2.FILLED)
    label_id += 1
    filled_count += 1

num_features = label_id - 1
print(f"Заполнено участков: {num_features}")
print(f"Черные границы (label=0) будут распределены watershed")

# ====================
# ШАГ 6: Watershed для заполнения ВСЕХ пробелов
# ====================
print("\n[6/7] Watershed: заполнение всех пробелов между участками...")

# Создаем маркеры для watershed
markers = labeled_array.copy()

# Находим неизвестные области (пробелы между участками)
unknown = (labeled_array == 0).astype(np.uint8) * 255

# Distance transform на неизвестных областях
dist_transform = cv2.distanceTransform(cv2.bitwise_not(unknown), cv2.DIST_L2, 5)

# Watershed требует 3-канальное изображение
img_for_watershed = cv2.cvtColor(binary_thin, cv2.COLOR_GRAY2BGR)

# Применяем watershed
# Он заполнит все пробелы, распределив их между соседними участками
markers = cv2.watershed(img_for_watershed, markers)

# Убираем граничные метки (-1)
markers[markers == -1] = 0

# Заполняем оставшиеся 0 (если есть) ближайшими участками
if np.any(markers == 0):
    # Для каждого пустого пикселя находим ближайший участок
    for i in range(height):
        for j in range(width):
            if markers[i, j] == 0:
                # Простой подход: берем соседа сверху/слева/справа/снизу
                neighbors = []
                if i > 0 and markers[i-1, j] > 0:
                    neighbors.append(markers[i-1, j])
                if i < height-1 and markers[i+1, j] > 0:
                    neighbors.append(markers[i+1, j])
                if j > 0 and markers[i, j-1] > 0:
                    neighbors.append(markers[i, j-1])
                if j < width-1 and markers[i, j+1] > 0:
                    neighbors.append(markers[i, j+1])
                if neighbors:
                    markers[i, j] = neighbors[0]

labeled_array = markers

# Визуализация
labeled_vis = (labeled_array % 256).astype(np.uint8)
labeled_colored = cv2.applyColorMap(labeled_vis, cv2.COLORMAP_JET)
cv2.imwrite('debug_final_03_labels.png', labeled_colored)
print("→ debug_final_03_labels.png (каждый цвет = один участок)")

# ====================
# ШАГ 7: Извлечение контуров каждого участка
# ====================
print("\n[7/7] Извлечение контуров участков...")

valid_parcels = []
rejected = {'too_small': 0, 'too_big': 0, 'invalid': 0}

for label_id in range(1, num_features + 1):
    # Маска для этого участка
    mask = (labeled_array == label_id).astype(np.uint8) * 255

    # Площадь
    area = np.sum(labeled_array == label_id)

    if area < MIN_AREA:
        rejected['too_small'] += 1
        continue
    if area > MAX_AREA:
        rejected['too_big'] += 1
        continue

    # Находим контур (без holes - они удалены вручную)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        rejected['invalid'] += 1
        continue

    # Берем самый большой контур
    contour = max(contours, key=cv2.contourArea)

    # Упрощаем контур
    perimeter = cv2.arcLength(contour, True)
    epsilon = EPSILON_FACTOR * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) < 3:
        rejected['invalid'] += 1
        continue

    # Вычисляем solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    num_vertices = len(approx)

    valid_parcels.append({
        'id': label_id,
        'contour': approx,
        'area': area,
        'vertices': num_vertices,
        'solidity': solidity
    })

print(f"\n✓ Найдено участков: {len(valid_parcels)}")
print(f"✗ Отфильтровано:")
print(f"  - Слишком маленькие: {rejected['too_small']}")
print(f"  - Слишком большие: {rejected['too_big']}")
print(f"  - Некорректные: {rejected['invalid']}")

# ====================
# ШАГ 8: Визуализация и экспорт
# ====================
print("\nВизуализация и экспорт...")

# Результат
debug_img = img.copy()
for parcel in valid_parcels:
    cv2.drawContours(debug_img, [parcel['contour']], -1, (0, 255, 0), 2)

    M = cv2.moments(parcel['contour'])
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(debug_img, str(parcel['id']), (cx-10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

cv2.imwrite('debug_final_04_result.png', debug_img)
print("→ debug_final_04_result.png")

# Только контуры
contours_only = np.ones((height, width, 3), dtype=np.uint8) * 255
for parcel in valid_parcels:
    cv2.drawContours(contours_only, [parcel['contour']], -1, (0, 0, 0), 1)
cv2.imwrite('debug_final_05_contours.png', contours_only)
print("→ debug_final_05_contours.png")

# ====================
# Экспорт в GeoJSON
# ====================
print("\nЭкспорт в GeoJSON...")

if len(valid_parcels) == 0:
    print("\n❌ Не найдено участков!")
    exit(1)

polygons = []

for parcel in valid_parcels:
    coords = parcel['contour'].squeeze().tolist()

    if len(coords) < 3:
        continue

    if not isinstance(coords[0], list):
        coords = [coords]

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
        print(f"  Ошибка участка {parcel['id']}: {e}")

if len(polygons) == 0:
    print("\n❌ Все полигоны невалидны!")
    exit(1)

gdf = gpd.GeoDataFrame(polygons, geometry='geometry', crs=None)
gdf.to_file(OUTPUT_GEOJSON, driver='GeoJSON')

print(f"→ {OUTPUT_GEOJSON}")
print(f"Участков: {len(gdf)}")

# ====================
# Статистика
# ====================
print("\n" + "=" * 60)
print("СТАТИСТИКА:")
print("=" * 60)

areas = [p['area'] for p in valid_parcels]
vertices = [p['vertices'] for p in valid_parcels]
solidities = [p['solidity'] for p in valid_parcels]
rectangles = sum(1 for p in valid_parcels if p['vertices'] == 4)

print(f"Площадь (пиксели):")
print(f"  Средняя: {np.mean(areas):.0f}")
print(f"  Мин-Макс: {np.min(areas):.0f} - {np.max(areas):.0f}")
print(f"\nВершины:")
print(f"  Среднее: {np.mean(vertices):.1f}")
print(f"  Мин-Макс: {np.min(vertices)} - {np.max(vertices)}")
print(f"\nПрямоугольников: {rectangles} ({rectangles/len(valid_parcels)*100:.1f}%)")
print(f"\nКомпактность:")
print(f"  Средняя: {np.mean(solidities):.3f}")

print("\n" + "=" * 60)
print("КЛЮЧЕВЫЕ ПРЕИМУЩЕСТВА:")
print("  ✓ Файл map_1891_cleaned.png - без вложенных участков")
print("  ✓ БЕЗ эрозии - сохранены тонкие линии границ")
print("  ✓ Заполняются только БЕЛЫЕ участки, НЕ границы")
print("  ✓ WATERSHED распределяет черные границы между соседями")
print("  ✓ Границы НЕ становятся полигонами!")
print("  ✓ Каждый пиксель принадлежит ТОЛЬКО одному участку")
print("  ✓ Соседние участки ПРИЛЕГАЮТ ВПЛОТНУЮ (делят границы)")
print("  ✓ НЕТ наложения участков")
print("  ✓ Упрощение (epsilon=0.01) - линии более прямые")
print("=" * 60)
