import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import nearest_points

# ====================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ
# ====================
def point_to_segment_distance(point, seg_start, seg_end):
    """
    Вычисляет расстояние от точки до отрезка и ближайшую точку на отрезке.

    Returns:
        distance: float - расстояние
        closest_point: np.array - ближайшая точка на отрезке
    """
    # Вектор отрезка
    segment_vec = seg_end - seg_start
    segment_len_sq = np.dot(segment_vec, segment_vec)

    # Если отрезок вырожденный (точка)
    if segment_len_sq < 1e-6:
        return np.linalg.norm(point - seg_start), seg_start

    # Проекция точки на прямую, содержащую отрезок
    # t = 0 -> seg_start, t = 1 -> seg_end
    t = np.dot(point - seg_start, segment_vec) / segment_len_sq

    # Ограничиваем t отрезком [0, 1]
    t = np.clip(t, 0.0, 1.0)

    # Ближайшая точка на отрезке
    closest = seg_start + t * segment_vec

    # Расстояние
    distance = np.linalg.norm(point - closest)

    return distance, closest

# ====================
# ПАРАМЕТРЫ
# ====================
INPUT_IMAGE = 'map_1891_cleaned.png'
OUTPUT_GEOJSON = 'map_1891_raw.geojson'

# Фильтры
MIN_AREA = 30
MAX_AREA = 100000

# Упрощение контуров
EPSILON_FACTOR = 0.007  # 0.7% от периметра

# Edge snapping - приближение к ребрам
EDGE_SNAP_TOLERANCE = 8.0  # Пиксели - вершина привязывается к ребру если ближе этого расстояния

print("=" * 60)
print("ИЗВЛЕЧЕНИЕ УЧАСТКОВ С EDGE-BASED SNAPPING")
print("=" * 60)
print(f"Входной файл: {INPUT_IMAGE}")
print(f"Упрощение: epsilon={EPSILON_FACTOR}")
print(f"Edge snap tolerance: {EDGE_SNAP_TOLERANCE} пикселей")
print("\nЛОГИКА: Вершина привязывается к ребру только если она")
print("        находится БЛИЗКО к линии границы соседа")

# ====================
# ШАГ 1: Загрузка и бинаризация
# ====================
print("\n[1/5] Загрузка и бинаризация...")
img = cv2.imread(INPUT_IMAGE)
if img is None:
    raise FileNotFoundError(f"Файл {INPUT_IMAGE} не найден!")

height, width = img.shape[:2]
print(f"Размер: {width}x{height} пикселей")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
cv2.imwrite('debug_edge_01_binary.png', binary)
print("→ debug_edge_01_binary.png")

# ====================
# ШАГ 2: Поиск контуров
# ====================
print("\n[2/5] Поиск контуров белых участков...")
contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
print(f"Найдено контуров: {len(contours)}")

# ====================
# ШАГ 3: Фильтрация и упрощение
# ====================
print("\n[3/5] Фильтрация и упрощение контуров...")

valid_parcels = []
rejected = {'too_small': 0, 'too_big': 0, 'invalid': 0}

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)

    if area < MIN_AREA:
        rejected['too_small'] += 1
        continue
    if area > MAX_AREA:
        rejected['too_big'] += 1
        continue

    perimeter = cv2.arcLength(contour, True)
    epsilon = EPSILON_FACTOR * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) < 3:
        rejected['invalid'] += 1
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
# ШАГ 4: EDGE-BASED SNAPPING (ключевой шаг!)
# ====================
print(f"\n[4/5] Edge-Based Snapping (tolerance={EDGE_SNAP_TOLERANCE} пикселей)...")
print("Привязка вершин к ребрам соседних участков...")

snap_count = 0
max_iterations = 5  # Максимум итераций для сходимости

for iteration in range(max_iterations):
    iteration_snaps = 0
    print(f"\n  Итерация {iteration + 1}/{max_iterations}...")

    # Для каждой пары полигонов
    for i, parcel_a in enumerate(valid_parcels):
        points_a = parcel_a['points']

        for j in range(i + 1, len(valid_parcels)):
            parcel_b = valid_parcels[j]
            points_b = parcel_b['points']

            # Проверяем вершины A против ребер B
            for idx_a in range(len(points_a)):
                va = np.array(points_a[idx_a])

                # Проходим по всем ребрам полигона B
                for idx_b in range(len(points_b)):
                    # Ребро от точки idx_b до следующей
                    edge_start = np.array(points_b[idx_b])
                    edge_end = np.array(points_b[(idx_b + 1) % len(points_b)])

                    # Вычисляем расстояние от вершины до ребра
                    dist, closest_point = point_to_segment_distance(va, edge_start, edge_end)

                    # Если вершина близко к ребру - привязываем
                    if dist < EDGE_SNAP_TOLERANCE and dist > 0.1:
                        # Перемещаем вершину к ближайшей точке на ребре
                        points_a[idx_a] = closest_point.tolist()
                        iteration_snaps += 1
                        snap_count += 1

            # Проверяем вершины B против ребер A
            for idx_b in range(len(points_b)):
                vb = np.array(points_b[idx_b])

                # Проходим по всем ребрам полигона A
                for idx_a in range(len(points_a)):
                    edge_start = np.array(points_a[idx_a])
                    edge_end = np.array(points_a[(idx_a + 1) % len(points_a)])

                    dist, closest_point = point_to_segment_distance(vb, edge_start, edge_end)

                    if dist < EDGE_SNAP_TOLERANCE and dist > 0.1:
                        points_b[idx_b] = closest_point.tolist()
                        iteration_snaps += 1
                        snap_count += 1

    print(f"    Привязано вершин: {iteration_snaps}")

    # Если на итерации ничего не изменилось - выходим
    if iteration_snaps == 0:
        print(f"  Сходимость достигнута на итерации {iteration + 1}")
        break

print(f"\nВсего привязано вершин: {snap_count}")

# ====================
# ШАГ 5: Визуализация и экспорт
# ====================
print("\n[5/5] Визуализация и экспорт в GeoJSON...")

# Визуализация
debug_img = img.copy()
for parcel in valid_parcels:
    points_np = np.array(parcel['points'], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(debug_img, [points_np], True, (0, 255, 0), 2)

    M = cv2.moments(points_np)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(debug_img, str(parcel['id']), (cx-10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

cv2.imwrite('debug_edge_02_result.png', debug_img)
print("→ debug_edge_02_result.png")

# Только контуры
contours_only = np.ones((height, width, 3), dtype=np.uint8) * 255
for parcel in valid_parcels:
    points_np = np.array(parcel['points'], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(contours_only, [points_np], True, (0, 0, 0), 1)
cv2.imwrite('debug_edge_03_contours.png', contours_only)
print("→ debug_edge_03_contours.png")

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

print(f"Участков: {len(valid_parcels)}")
print(f"\nПлощадь (пиксели):")
print(f"  Средняя: {np.mean(areas):.0f}")
print(f"  Мин-Макс: {np.min(areas):.0f} - {np.max(areas):.0f}")
print(f"\nВершины:")
print(f"  Среднее: {np.mean(vertices):.1f}")
print(f"  Мин-Макс: {np.min(vertices)} - {np.max(vertices)}")
print(f"\nEdge-Based Snapping:")
print(f"  Привязано вершин: {snap_count}")
print(f"  Tolerance: {EDGE_SNAP_TOLERANCE} пикселей")

print("\n" + "=" * 60)
print("ПРЕИМУЩЕСТВА EDGE-BASED ПОДХОДА:")
print("  ✓ Привязка ТОЛЬКО к реальным общим границам")
print("  ✓ Полигоны через дорогу НЕ объединяются")
print("  ✓ Вершина привязывается к линии границы, а не к соседней вершине")
print("  ✓ Работает лучше для маленьких участков")
print("  ✓ Не расширяет полигоны во все стороны")
print("=" * 60)
