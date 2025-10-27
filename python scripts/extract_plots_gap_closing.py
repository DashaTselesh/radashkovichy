import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree

# ====================
# ПАРАМЕТРЫ
# ====================
INPUT_IMAGE = 'map_1891_cleaned.png'
OUTPUT_GEOJSON = 'map_1891_raw.geojson'

# Фильтры
MIN_AREA = 30
MAX_AREA = 100000

# Сглаживание перед скелетизацией
BLUR_KERNEL = 3  # Размер kernel для Gaussian blur
CLOSING_KERNEL = 3  # Размер kernel для morphological closing

# Gap closing
GAP_THRESHOLD = 7  # Максимальное расстояние для соединения разрывов (px)

# Упрощение контуров
EPSILON_FACTOR = 0.003  # 0.3% от периметра (уменьшено для меньшего упрощения)

print("=" * 60)
print("GAP CLOSING ALGORITHM - УМНОЕ СОЕДИНЕНИЕ РАЗРЫВОВ")
print("=" * 60)
print(f"Входной файл: {INPUT_IMAGE}")
print(f"Blur kernel: {BLUR_KERNEL}x{BLUR_KERNEL}")
print(f"Closing kernel: {CLOSING_KERNEL}x{CLOSING_KERNEL}")
print(f"Gap threshold: {GAP_THRESHOLD}px")
print(f"\nАЛГОРИТМ:")
print(f"  1. Сглаживание черных линий (blur + closing)")
print(f"  2. Скелетизация → 1px по центру")
print(f"  3. Поиск endpoints (концы линий)")
print(f"  4. Соединение разрывов < {GAP_THRESHOLD}px")
print(f"  5. Замкнутые линии → полигоны участков")

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
# ШАГ 2: СГЛАЖИВАНИЕ ДО СКЕЛЕТИЗАЦИИ
# ====================
print(f"\n[2/7] Сглаживание линий (ДО скелетизации)...")

# Слабое Gaussian blur
print(f"  → Gaussian blur (kernel {BLUR_KERNEL}x{BLUR_KERNEL})...")
blurred = cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)
cv2.imwrite('debug_gap_01_blurred.png', blurred)
print("  → debug_gap_01_blurred.png")

# Бинаризация после blur
_, binary_white = cv2.threshold(blurred, 250, 255, cv2.THRESH_BINARY)
cv2.imwrite('debug_gap_02_binary.png', binary_white)
print("  → debug_gap_02_binary.png")

# Инвертируем для работы с черными линиями
binary_black = cv2.bitwise_not(binary_white)

# Morphological closing - заполняет микродыры в черных линиях
print(f"  → Closing черных линий (kernel {CLOSING_KERNEL}x{CLOSING_KERNEL})...")
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSING_KERNEL, CLOSING_KERNEL))
binary_black_closed = cv2.morphologyEx(binary_black, cv2.MORPH_CLOSE, kernel_close)

cv2.imwrite('debug_gap_03_black_closed.png', binary_black_closed)
print("  → debug_gap_03_black_closed.png (сглаженные черные линии)")

# ====================
# ШАГ 3: СКЕЛЕТИЗАЦИЯ
# ====================
print("\n[3/7] Скелетизация сглаженных линий...")

# Скелетизация
binary_bool = binary_black_closed > 0
skeleton_bool = skeletonize(binary_bool)
skeleton = (skeleton_bool * 255).astype(np.uint8)

cv2.imwrite('debug_gap_04_skeleton.png', skeleton)
print("→ debug_gap_04_skeleton.png (скелет 1px)")

# Статистика
original_pixels = np.sum(binary_black_closed > 0)
skeleton_pixels = np.sum(skeleton > 0)
reduction = 100 * (1 - skeleton_pixels / original_pixels)
print(f"Уменьшение: {original_pixels} → {skeleton_pixels} пикселей ({reduction:.1f}%)")

# ====================
# ШАГ 4: ПОИСК ENDPOINTS (концы линий)
# ====================
print(f"\n[4/7] Поиск endpoints (концы линий с разрывами)...")

# Функция для подсчета соседей пикселя
def count_neighbors(binary_img, y, x):
    """Подсчитывает количество белых соседей пикселя в окне 3x3"""
    neighbors = 0
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < binary_img.shape[0] and 0 <= nx < binary_img.shape[1]:
                if binary_img[ny, nx] > 0:
                    neighbors += 1
    return neighbors

# Находим все точки скелета
skeleton_points = np.argwhere(skeleton > 0)

# Классифицируем точки
endpoints = []
junction_points = []
regular_points = []

print("Анализ точек скелета...")
for point in skeleton_points:
    y, x = point
    neighbors = count_neighbors(skeleton, y, x)

    if neighbors == 1:
        endpoints.append((x, y))  # Endpoint - конец линии
    elif neighbors >= 3:
        junction_points.append((x, y))  # Junction - развилка
    else:
        regular_points.append((x, y))  # Regular - обычная точка линии

print(f"Найдено:")
print(f"  - Endpoints (концы линий): {len(endpoints)}")
print(f"  - Junction points (развилки): {len(junction_points)}")
print(f"  - Regular points: {len(regular_points)}")

# Визуализация endpoints
endpoints_vis = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
for x, y in endpoints:
    cv2.circle(endpoints_vis, (x, y), 2, (0, 0, 255), -1)  # Красные точки
for x, y in junction_points:
    cv2.circle(endpoints_vis, (x, y), 2, (0, 255, 0), -1)  # Зеленые точки

cv2.imwrite('debug_gap_05_endpoints.png', endpoints_vis)
print("→ debug_gap_05_endpoints.png (красные=endpoints, зеленые=junctions)")

# ====================
# ШАГ 5: GAP CLOSING (соединение разрывов)
# ====================
print(f"\n[5/7] Gap Closing - соединение разрывов < {GAP_THRESHOLD}px...")

# Создаем копию скелета для рисования соединений
skeleton_closed = skeleton.copy()

# Строим KDTree для быстрого поиска ближайших точек
# Используем все точки скелета (не только endpoints)
all_skeleton_coords = [(x, y) for y, x in skeleton_points]
tree = cKDTree(all_skeleton_coords)

connections_made = 0
connection_lines = []

print("Поиск и соединение близких разрывов...")
for ep_x, ep_y in endpoints:
    # Ищем ближайшие точки скелета в радиусе GAP_THRESHOLD
    indices = tree.query_ball_point([ep_x, ep_y], GAP_THRESHOLD)

    if len(indices) > 1:  # Есть другие точки рядом (кроме самой себя)
        # Находим ближайшую точку (не саму себя)
        min_dist = float('inf')
        closest_point = None

        for idx in indices:
            target_x, target_y = all_skeleton_coords[idx]
            if target_x == ep_x and target_y == ep_y:
                continue  # Пропускаем саму себя

            dist = np.sqrt((target_x - ep_x)**2 + (target_y - ep_y)**2)
            if dist < min_dist and dist > 0.1:
                min_dist = dist
                closest_point = (target_x, target_y)

        # Если нашли близкую точку - рисуем соединительную линию
        if closest_point and min_dist <= GAP_THRESHOLD:
            cv2.line(skeleton_closed, (ep_x, ep_y), closest_point, 255, 1)
            connections_made += 1
            connection_lines.append(((ep_x, ep_y), closest_point))

print(f"Создано соединений: {connections_made}")

# Визуализация соединений
connections_vis = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
for (start, end) in connection_lines:
    cv2.line(connections_vis, start, end, (255, 0, 0), 1)  # Синие линии

cv2.imwrite('debug_gap_06_connections.png', connections_vis)
print("→ debug_gap_06_connections.png (синие=новые соединения)")

# Сохраняем замкнутый скелет
cv2.imwrite('debug_gap_07_skeleton_closed.png', skeleton_closed)
print("→ debug_gap_07_skeleton_closed.png (замкнутый скелет)")

# ====================
# ШАГ 6: СОЗДАНИЕ ПОЛИГОНОВ
# ====================
print("\n[6/7] Создание полигонов из замкнутого скелета...")

# skeleton_closed = белые линии на черном фоне
# НЕ инвертируем! Участки = черные области между белыми линиями
# findContours найдет черные замкнутые области = участки

cv2.imwrite('debug_gap_08_skeleton_for_contours.png', skeleton_closed)
print("→ debug_gap_08_skeleton_for_contours.png (белые границы, черные участки)")

# Инвертируем для визуализации
white_areas_vis = cv2.bitwise_not(skeleton_closed)
cv2.imwrite('debug_gap_08_white_areas.png', white_areas_vis)
print("→ debug_gap_08_white_areas.png (для визуализации - черные границы, белые участки)")

# Находим контуры ЧЕРНЫХ областей (участков) между БЕЛЫМИ линиями (границами)
# Инвертируем временно для findContours (он ищет белые контуры)
inverted_for_contours = cv2.bitwise_not(skeleton_closed)
all_contours, hierarchy = cv2.findContours(inverted_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Найдено контуров: {len(all_contours)}")

# Визуализация ВСЕХ найденных контуров (до фильтрации)
all_contours_vis = cv2.cvtColor(white_areas_vis, cv2.COLOR_GRAY2BGR)
cv2.drawContours(all_contours_vis, all_contours, -1, (0, 255, 0), 1)
cv2.imwrite('debug_gap_08a_all_contours.png', all_contours_vis)
print(f"→ debug_gap_08a_all_contours.png (все {len(all_contours)} контуров до фильтрации)")

# DEBUG: площади первых контуров
if len(all_contours) > 0:
    areas_debug = sorted([cv2.contourArea(c) for c in all_contours], reverse=True)[:10]
    print(f"Площади топ-10 контуров: {[int(a) for a in areas_debug]}")

# ====================
# Фильтрация и упрощение
# ====================
print("\n[7/7] Фильтрация и упрощение контуров...")

valid_parcels = []
rejected = {'too_small': 0, 'too_big': 0, 'invalid': 0}

for i, contour in enumerate(all_contours):
    area = cv2.contourArea(contour)

    # DEBUG для первых 10 контуров
    if i < 10:
        print(f"  Контур {i}: площадь={int(area)}, MIN={MIN_AREA}, MAX={MAX_AREA}", end="")

    # Фильтр по площади
    if area < MIN_AREA:
        rejected['too_small'] += 1
        if i < 10:
            print(" → too_small")
        continue
    if area > MAX_AREA:
        rejected['too_big'] += 1
        if i < 10:
            print(" → too_big")
        continue

    if i < 10:
        print(" → OK")

    # Упрощение контура
    perimeter = cv2.arcLength(contour, True)
    epsilon = EPSILON_FACTOR * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # DEBUG для первых контуров
    if i < 5:
        print(f"    Упрощение: {len(contour)} → {len(approx)} вершин")

    if len(approx) < 3:
        rejected['invalid'] += 1
        if i < 5:
            print(f"    → invalid (< 3 вершин)")
        continue

    points = approx.squeeze().tolist()

    if len(points) < 3:
        rejected['invalid'] += 1
        continue

    if not isinstance(points[0], list):
        points = [points]

    valid_parcels.append({
        'id': i,
        'points': points,
        'area': area
    })

print(f"\n✓ Найдено участков: {len(valid_parcels)}")
print(f"✗ Отфильтровано:")
print(f"  - Слишком маленькие: {rejected['too_small']}")
print(f"  - Слишком большие: {rejected['too_big']}")
print(f"  - Некорректные: {rejected['invalid']}")

# ====================
# Визуализация
# ====================
print("\nВизуализация...")

# Визуализация на белых участках
debug_img = cv2.cvtColor(white_areas_vis, cv2.COLOR_GRAY2BGR)
for parcel in valid_parcels:
    points_np = np.array(parcel['points'], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(debug_img, [points_np], True, (0, 255, 0), 2)

    M = cv2.moments(points_np)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(debug_img, str(parcel['id']), (cx-10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

cv2.imwrite('debug_gap_09_result.png', debug_img)
print("→ debug_gap_09_result.png (финальные контуры)")

# Только контуры
contours_only = np.ones((height, width, 3), dtype=np.uint8) * 255
for parcel in valid_parcels:
    points_np = np.array(parcel['points'], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(contours_only, [points_np], True, (0, 0, 0), 1)
cv2.imwrite('debug_gap_10_contours.png', contours_only)
print("→ debug_gap_10_contours.png")

# ====================
# Экспорт в GeoJSON
# ====================
print("\nЭкспорт в GeoJSON...")

polygons = []

for parcel in valid_parcels:
    coords = parcel['points'].copy()

    if coords[0] != coords[-1]:
        coords.append(coords[0])

    try:
        poly = Polygon(coords)

        if not poly.is_valid:
            print(f"  Предупреждение: участок {parcel['id']} невалиден, пропускаем")
            continue

        polygons.append({
            'geometry': poly,
            'auto_id': parcel['id'],
            'area_pixels': int(parcel['area']),
            'vertices': len(parcel['points']),
            'plot_number': None,
            'owner_first_name': None,
            'owner_last_name': None
        })
    except Exception as e:
        print(f"  Ошибка участка {parcel['id']}: {e}")

if len(polygons) == 0:
    print("\n❌ Не удалось создать ни одного валидного полигона!")
    exit(1)

gdf = gpd.GeoDataFrame(polygons, geometry='geometry', crs=None)
gdf.to_file(OUTPUT_GEOJSON, driver='GeoJSON')

print(f"→ {OUTPUT_GEOJSON}")
print(f"Участков в файле: {len(gdf)}")

# ====================
# Статистика
# ====================
print("\n" + "=" * 60)
print("СТАТИСТИКА:")
print("=" * 60)

areas = [p['area'] for p in valid_parcels]
vertices = [len(p['points']) for p in valid_parcels]

if len(areas) > 0:
    print(f"Участков: {len(valid_parcels)}")
    print(f"\nПлощадь (пиксели):")
    print(f"  Средняя: {np.mean(areas):.0f}")
    print(f"  Мин-Макс: {np.min(areas):.0f} - {np.max(areas):.0f}")
    print(f"\nВершины:")
    print(f"  Среднее: {np.mean(vertices):.1f}")
    print(f"  Мин-Макс: {np.min(vertices)} - {np.max(vertices)}")

print(f"\nGap Closing:")
print(f"  Endpoints найдено: {len(endpoints)}")
print(f"  Соединений создано: {connections_made}")
print(f"  Gap threshold: {GAP_THRESHOLD}px")

print("\n" + "=" * 60)
print("ПРЕИМУЩЕСТВА GAP CLOSING:")
print("  ✓ Сглаживание ДО скелетизации")
print("  ✓ Скелет 1px по центру любой толщины")
print("  ✓ Умное соединение только разрывов")
print("  ✓ НЕ расширяет существующие линии")
print("  ✓ Участки вплотную друг к другу")
print("  ✓ Граница точно по центру")
print("=" * 60)
