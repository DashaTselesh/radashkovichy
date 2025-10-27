import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

# ====================
# ПАРАМЕТРЫ
# ====================
INPUT_IMAGE = 'map_1891.png'  # Белый фон, черные границы
OUTPUT_GEOJSON = 'map_1891_raw.geojson'

# Фильтры
MIN_AREA = 30         # Минимальная площадь участка (пиксели)
MAX_AREA = 100000     # Максимальная площадь

# Прямоугольная аппроксимация (ослабленная)
RECTANGULARIZE = True
RECT_SOLIDITY_THRESHOLD = 0.85  # Только очень компактные → прямоугольник (было 0.7)

print("=" * 60)
print("ПРОСТОЕ ИЗВЛЕЧЕНИЕ УЧАСТКОВ")
print("=" * 60)
print(f"Входной файл: {INPUT_IMAGE}")
print(f"Ожидаемое количество участков: ~188")

# ====================
# ШАГ 1: Загрузка
# ====================
print("\n[1/4] Загрузка изображения...")
img = cv2.imread(INPUT_IMAGE)
if img is None:
    raise FileNotFoundError(f"Файл {INPUT_IMAGE} не найден!")

height, width = img.shape[:2]
print(f"Размер: {width}x{height} пикселей")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ====================
# ШАГ 2: Бинаризация и утончение границ
# ====================
print("\n[2/5] Бинаризация...")
# Простой порог: всё что не белое (< 250) считаем линией
_, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

cv2.imwrite('debug_simple_01_binary.png', binary)
print("→ Сохранено: debug_simple_01_binary.png")

# Утончение границ для сближения участков
print("\n[3/5] Утончение границ (участки станут ближе)...")
# Инверсия: границы = белые
binary_inv = cv2.bitwise_not(binary)

# Эрозия границ (делаем их тоньше)
kernel_erode = np.ones((2, 2), np.uint8)
boundaries_thin = cv2.erode(binary_inv, kernel_erode, iterations=2)

# Инверсия обратно: участки = белые
binary_thin = cv2.bitwise_not(boundaries_thin)

cv2.imwrite('debug_simple_02_thinned.png', binary_thin)
print("→ Сохранено: debug_simple_02_thinned.png")

# ====================
# ШАГ 4: Поиск контуров
# ====================
print("\n[4/5] Поиск контуров участков...")

# Находим все замкнутые контуры (используем утонченное изображение!)
contours, hierarchy = cv2.findContours(binary_thin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

print(f"Найдено контуров: {len(contours)}")

# ====================
# ШАГ 5: Фильтрация и упрощение контуров
# ====================
print("\n[5/5] Фильтрация и упрощение участков...")

valid_parcels = []
rejected = {'too_small': 0, 'too_big': 0, 'invalid': 0}

for i, contour in enumerate(contours):
    # Площадь
    area = cv2.contourArea(contour)

    # Фильтр по площади
    if area < MIN_AREA:
        rejected['too_small'] += 1
        continue
    if area > MAX_AREA:
        rejected['too_big'] += 1
        continue

    # Вычисляем solidity (компактность)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        rejected['invalid'] += 1
        continue

    solidity = area / hull_area

    # АГРЕССИВНОЕ упрощение контура для прямых линий
    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.015 * perimeter  # Увеличено с 0.01 до 0.015 (более прямые линии)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Минимум 3 точки для полигона
    if len(approx) < 3:
        rejected['invalid'] += 1
        continue

    # ОСЛАБЛЕННАЯ прямоугольная аппроксимация
    # Только для ОЧЕНЬ компактных фигур (solidity >= 0.85)
    if RECTANGULARIZE and solidity >= RECT_SOLIDITY_THRESHOLD:
        # Очень компактная фигура → делаем прямоугольником
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        final_contour = box.reshape(-1, 1, 2)
        num_vertices = 4
    else:
        # Оставляем упрощенный контур (с прямыми линиями, но не обязательно прямоугольник)
        final_contour = approx
        num_vertices = len(approx)

    valid_parcels.append({
        'id': i,
        'contour': final_contour,
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
# Визуализация
# ====================
print("\nСоздание визуализации...")

# Результат с номерами
debug_img = img.copy()
for parcel in valid_parcels:
    cv2.drawContours(debug_img, [parcel['contour']], -1, (0, 255, 0), 3)

    M = cv2.moments(parcel['contour'])
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(debug_img, str(parcel['id']), (cx-15, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

cv2.imwrite('debug_simple_03_result.png', debug_img)
print("→ Сохранено: debug_simple_03_result.png")

# Только контуры
contours_only = np.ones((height, width, 3), dtype=np.uint8) * 255
for parcel in valid_parcels:
    cv2.drawContours(contours_only, [parcel['contour']], -1, (0, 0, 0), 2)
cv2.imwrite('debug_simple_04_contours.png', contours_only)
print("→ Сохранено: debug_simple_04_contours.png")

# ====================
# Экспорт в GeoJSON
# ====================
print("\nЭкспорт в GeoJSON...")

if len(valid_parcels) == 0:
    print("\n❌ ОШИБКА: Не найдено ни одного участка!")
    print("Проверьте:")
    print("  1. Файл map_1891.png существует")
    print("  2. Белый фон (255), черные линии границ (0-10)")
    print("  3. Все контуры замкнуты")
    exit(1)

polygons = []

for parcel in valid_parcels:
    coords = parcel['contour'].squeeze().tolist()

    if len(coords) < 3:
        continue

    if not isinstance(coords[0], list):
        coords = [coords]

    # Замыкание полигона
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

if len(polygons) == 0:
    print("\n❌ Все полигоны оказались невалидными!")
    exit(1)

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

areas = [p['area'] for p in valid_parcels]
vertices = [p['vertices'] for p in valid_parcels]
solidities = [p['solidity'] for p in valid_parcels]
rectangles = sum(1 for p in valid_parcels if p['vertices'] == 4)

print(f"Площадь (пиксели):")
print(f"  Средняя: {np.mean(areas):.0f}")
print(f"  Минимум: {np.min(areas):.0f}")
print(f"  Максимум: {np.max(areas):.0f}")
print(f"\nВершины:")
print(f"  Среднее: {np.mean(vertices):.1f}")
print(f"  Минимум: {np.min(vertices)}")
print(f"  Максимум: {np.max(vertices)}")
print(f"\nПрямоугольников: {rectangles} ({rectangles/len(valid_parcels)*100:.1f}%)")
print(f"\nКомпактность:")
print(f"  Средняя: {np.mean(solidities):.3f}")
print(f"  Минимум: {np.min(solidities):.3f}")

print("=" * 60)
print("\n✓ ГОТОВО!")
print("=" * 60)
