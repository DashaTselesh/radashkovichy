import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import json

# ====================
# ПАРАМЕТРЫ (настроенные под ваш план)
# ====================
INPUT_IMAGE = 'map_1891.png'
OUTPUT_GEOJSON = 'map_1891_raw.geojson'

# Бинаризация
THRESHOLD_VALUE = 127

# Фильтр по площади (в пикселях)
MIN_AREA = 100        # Уменьшено с 500 - захватим очень мелкие участки
MAX_AREA = 1000000    # Исключим огромные артефакты

# Фильтр по компактности (solidity)
# Мусор = длинные линии с малой площадью (solidity < 0.3)
# Участки = компактные фигуры (solidity > 0.3)
MIN_SOLIDITY = 0.3

# Фильтр по количеству вершин
MIN_VERTICES = 4      # Минимум 4 точки для участка

# Упрощение контуров (агрессивное для прямоугольных участков)
EPSILON_FACTOR = 0.01  # Увеличено с 0.003 для более прямых линий

print("=" * 60)
print("ИЗВЛЕЧЕНИЕ УЧАСТКОВ ИЗ ПЛАНА РАДАШКОВИЧЕЙ")
print("=" * 60)
print(f"Ожидаемое количество участков: ~188")

# ====================
# ШАГ 1: Загрузка
# ====================
print("\n[1/8] Загрузка изображения...")
img = cv2.imread(INPUT_IMAGE)
if img is None:
    raise FileNotFoundError(f"Файл {INPUT_IMAGE} не найден!")

height, width = img.shape[:2]
print(f"Размер: {width}x{height} пикселей")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ====================
# ШАГ 2: Адаптивная бинаризация
# ====================
print("\n[2/8] Адаптивная бинаризация...")
# Лучше работает с неоднородным фоном
binary = cv2.adaptiveThreshold(
    gray, 
    255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY_INV,
    blockSize=15,
    C=5
)

cv2.imwrite('debug_01_adaptive.png', binary)
print("→ Сохранено: debug_01_adaptive.png")

# ====================
# ШАГ 3: Морфология - АГРЕССИВНОЕ замыкание разрывов
# ====================
print("\n[3/8] Замыкание разрывов в границах...")
# Большое ядро для закрытия разрывов
kernel_close = np.ones((5, 5), np.uint8)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=4)

cv2.imwrite('debug_02_closed.png', closed)
print("→ Сохранено: debug_02_closed.png (разрывы замкнуты)")

# НОВЫЙ ПОДХОД: Утончаем границы до 1 пикселя
# Это уберет расстояние между участками
print("\n[3.5/8] Утончение границ до 1 пикселя...")
# Эрозия границ (сделать их тоньше)
kernel_erode = np.ones((2, 2), np.uint8)
thinned = cv2.erode(closed, kernel_erode, iterations=1)
cv2.imwrite('debug_03_thinned.png', thinned)
print("→ Сохранено: debug_03_thinned.png (границы утончены)")

# Инвертируем изображение
# На оригинале: черные линии = границы, белый фон = участки
# После инверсии: белые линии = границы, черные области = участки
print("\n[3.6/8] Инверсия для извлечения участков...")
inverted = cv2.bitwise_not(thinned)
cv2.imwrite('debug_04_inverted.png', inverted)
print("→ Сохранено: debug_04_inverted.png (инвертировано)")

# НЕ используем MORPH_OPEN - он удаляет мелкие участки!
# Используем inverted напрямую
cleaned = inverted

# ====================
# ШАГ 4: Поиск контуров (заполненных участков)
# ====================
print("\n[4/8] Поиск контуров участков...")
# Теперь ищем БЕЛЫЕ области (участки), а не границы
contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
print(f"Найдено контуров: {len(contours)}")

# ====================
# ШАГ 5: Умная фильтрация
# ====================
print("\n[5/8] Фильтрация участков...")
valid_parcels = []
rejected = {'too_small': 0, 'too_big': 0, 'not_compact': 0, 'too_few_vertices': 0}

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    
    # Фильтр 1: площадь
    if area < MIN_AREA:
        rejected['too_small'] += 1
        continue
    if area > MAX_AREA:
        rejected['too_big'] += 1
        continue
    
    # Фильтр 2: компактность (исключаем длинные линии-мусор)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        solidity = area / hull_area
        if solidity < MIN_SOLIDITY:
            rejected['not_compact'] += 1
            continue
    
    # Упрощение контура
    perimeter = cv2.arcLength(contour, True)
    epsilon = EPSILON_FACTOR * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Фильтр 3: минимум вершин
    num_vertices = len(approx)
    if num_vertices < MIN_VERTICES:
        rejected['too_few_vertices'] += 1
        continue

    # Дополнительное упрощение: если участок похож на прямоугольник (4-6 вершин),
    # попробуем заменить на минимальный ограничивающий прямоугольник
    if 4 <= num_vertices <= 6 and solidity > 0.8:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        approx = box.reshape(-1, 1, 2)
    
    valid_parcels.append({
        'id': i,
        'contour': approx,
        'area': area,
        'vertices': num_vertices,
        'solidity': solidity
    })

print(f"\n✓ Найдено участков: {len(valid_parcels)}")
print(f"✗ Отфильтровано:")
print(f"  - Слишком маленькие: {rejected['too_small']}")
print(f"  - Слишком большие: {rejected['too_big']}")
print(f"  - Некомпактные (линии): {rejected['not_compact']}")
print(f"  - Мало вершин: {rejected['too_few_vertices']}")

# ====================
# ШАГ 6: Визуализация найденных участков
# ====================
print("\n[6/8] Создание визуализации...")
debug_img = img.copy()

for parcel in valid_parcels:
    # Контур зеленым
    cv2.drawContours(debug_img, [parcel['contour']], -1, (0, 255, 0), 4)
    
    # ID в центре
    M = cv2.moments(parcel['contour'])
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # ID красным
        cv2.putText(debug_img, str(parcel['id']), (cx-30, cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

cv2.imwrite('debug_05_contours.png', debug_img)
print("→ Сохранено: debug_05_contours.png")

# Дополнительная визуализация - только контуры
contours_only = np.ones((height, width, 3), dtype=np.uint8) * 255
for parcel in valid_parcels:
    cv2.drawContours(contours_only, [parcel['contour']], -1, (0, 0, 0), 2)
cv2.imwrite('debug_06_contours_only.png', contours_only)
print("→ Сохранено: debug_06_contours_only.png")

# ====================
# ШАГ 7: Конвертация в GeoJSON
# ====================
print("\n[7/8] Конвертация в GeoJSON...")
polygons = []

for parcel in valid_parcels:
    coords = parcel['contour'].squeeze().tolist()
    
    # Проверка
    if len(coords) < 3:
        continue
    
    # Обработка одномерных координат (если всего 1 точка стала списком)
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
# ШАГ 8: Детальная статистика
# ====================
print("\n[8/8] Статистика по участкам:")
print("=" * 60)
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
print("  1. debug_01_adaptive.png - адаптивная бинаризация")
print("  2. debug_02_closed.png - замкнутые разрывы")
print("  3. debug_03_thinned.png - утонченные границы (1-2 пикселя)")
print("  4. debug_04_inverted.png - инвертированное (участки = белые)")
print("  5. debug_05_contours.png - найденные участки на плане")
print("  6. debug_06_contours_only.png - только контуры")
print("  7. map_1891_raw.geojson - результат для QGIS")
print("\nЕсли найдено меньше/больше 188 участков - подкорректируем параметры!")
print("=" * 60)