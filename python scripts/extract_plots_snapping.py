import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from scipy.spatial import distance

# ====================
# ПАРАМЕТРЫ
# ====================
INPUT_IMAGE = 'map_1891_cleaned.png'
OUTPUT_GEOJSON = 'map_1891_raw.geojson'

# Фильтры
MIN_AREA = 30
MAX_AREA = 100000

# Упрощение контуров (умеренное - линии ровнее, но не идеально прямые)
EPSILON_FACTOR = 0.007  # 0.7% от периметра - умеренное сглаживание

# Vertex snapping (привязка вершин)
SNAP_TOLERANCE = 20.0  # Пиксели - вершины ближе этого расстояния будут объединены (увеличено для границ)

print("=" * 60)
print("ИЗВЛЕЧЕНИЕ УЧАСТКОВ С VERTEX SNAPPING")
print("=" * 60)
print(f"Входной файл: {INPUT_IMAGE}")
print(f"Упрощение: epsilon={EPSILON_FACTOR} (умеренное)")
print(f"Snap tolerance: {SNAP_TOLERANCE} пикселей")

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

# Бинаризация
_, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
cv2.imwrite('debug_snap_01_binary.png', binary)
print("→ debug_snap_01_binary.png")

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

    # Фильтр по площади
    if area < MIN_AREA:
        rejected['too_small'] += 1
        continue
    if area > MAX_AREA:
        rejected['too_big'] += 1
        continue

    # Упрощение контура (умеренное)
    perimeter = cv2.arcLength(contour, True)
    epsilon = EPSILON_FACTOR * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) < 3:
        rejected['invalid'] += 1
        continue

    # Преобразуем в список точек (x, y)
    points = approx.squeeze().tolist()

    # Проверка формата
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
# ШАГ 4: VERTEX SNAPPING (ключевой шаг!)
# ====================
print(f"\n[4/5] Vertex Snapping (tolerance={SNAP_TOLERANCE} пикселей)...")
print("Выравнивание вершин соседних участков...")

snap_count = 0

# Для каждой пары полигонов
for i, parcel_a in enumerate(valid_parcels):
    for j in range(i + 1, len(valid_parcels)):
        parcel_b = valid_parcels[j]

        points_a = parcel_a['points']
        points_b = parcel_b['points']

        # Для каждой вершины в A
        for idx_a in range(len(points_a)):
            va = np.array(points_a[idx_a])

            # Для каждой вершины в B
            for idx_b in range(len(points_b)):
                vb = np.array(points_b[idx_b])

                # Расстояние между вершинами
                dist = np.linalg.norm(va - vb)

                # Если близко - объединяем в среднюю точку
                if dist < SNAP_TOLERANCE and dist > 0:
                    mid_point = ((va + vb) / 2).tolist()

                    # Заменяем обе вершины на среднюю
                    points_a[idx_a] = mid_point
                    points_b[idx_b] = mid_point
                    snap_count += 1

print(f"Объединено вершин: {snap_count}")

# ====================
# ШАГ 5: Визуализация и экспорт
# ====================
print("\n[5/5] Визуализация и экспорт в GeoJSON...")

# Визуализация
debug_img = img.copy()
for parcel in valid_parcels:
    points_np = np.array(parcel['points'], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(debug_img, [points_np], True, (0, 255, 0), 2)

    # Центр для ID
    M = cv2.moments(points_np)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(debug_img, str(parcel['id']), (cx-10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

cv2.imwrite('debug_snap_02_result.png', debug_img)
print("→ debug_snap_02_result.png")

# Только контуры
contours_only = np.ones((height, width, 3), dtype=np.uint8) * 255
for parcel in valid_parcels:
    points_np = np.array(parcel['points'], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(contours_only, [points_np], True, (0, 0, 0), 1)
cv2.imwrite('debug_snap_03_contours.png', contours_only)
print("→ debug_snap_03_contours.png")

# ====================
# Экспорт в GeoJSON
# ====================
print("\nЭкспорт в GeoJSON...")

polygons = []

for parcel in valid_parcels:
    coords = parcel['points'].copy()

    # Замыкание полигона
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
print(f"\nVertex Snapping:")
print(f"  Объединено вершин: {snap_count}")
print(f"  Tolerance: {SNAP_TOLERANCE} пикселей")

print("\n" + "=" * 60)
print("ПРЕИМУЩЕСТВА ЭТОГО ПОДХОДА:")
print("  ✓ Простой и понятный алгоритм")
print("  ✓ Контуры максимально близки к оригиналу")
print("  ✓ Умеренное упрощение - линии ровнее, но не идеально прямые")
print("  ✓ Vertex Snapping - соседние участки делят общие вершины")
print("  ✓ Участки прилегают вплотную где они близко")
print("  ✓ НЕТ watershed артефактов")
print("  ✓ НЕ создаются полигоны из границ")
print("=" * 60)
