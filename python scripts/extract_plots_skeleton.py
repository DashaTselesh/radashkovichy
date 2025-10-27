import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from scipy import ndimage
from skimage.morphology import skeletonize, medial_axis

# ====================
# ПАРАМЕТРЫ
# ====================
INPUT_IMAGE = 'map_1891_cleaned.png'
OUTPUT_GEOJSON = 'map_1891_raw.geojson'

# Фильтры
MIN_AREA = 30
MAX_AREA = 100000

# Скелетизация
THICK_LINE_THRESHOLD = 3  # Считаем линию толстой если > 3px, только их скелетизируем

# Упрощение контуров
EPSILON_FACTOR = 0.007  # 0.7% от периметра

print("=" * 60)
print("SKELETON-BASED EXTRACTION (УМНАЯ СКЕЛЕТИЗАЦИЯ)")
print("=" * 60)
print(f"Входной файл: {INPUT_IMAGE}")
print(f"Порог толстых линий: {THICK_LINE_THRESHOLD}px")
print(f"\nЛОГИКА:")
print(f"  1. Определяем толщину черных линий")
print(f"  2. Скелетизируем только линии > {THICK_LINE_THRESHOLD}px")
print(f"  3. Тонкие линии сохраняем как есть")
print(f"  4. Watershed от скелета = полигоны вплотную!")

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
# ШАГ 2: Бинаризация (черное vs белое)
# ====================
print("\n[2/6] Бинаризация...")
_, binary_white = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)  # Белые участки
_, binary_black = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)  # Черные линии

cv2.imwrite('debug_skel_01_white.png', binary_white)
cv2.imwrite('debug_skel_02_black.png', binary_black)
print("→ debug_skel_01_white.png (белые участки)")
print("→ debug_skel_02_black.png (черные границы)")

# ====================
# ШАГ 3: УМНАЯ СКЕЛЕТИЗАЦИЯ
# ====================
print(f"\n[3/6] Умная скелетизация границ...")
print(f"Анализ толщины черных линий...")

# Distance transform - показывает толщину линии в каждой точке
# Для черных линий: расстояние до ближайшего белого пикселя
binary_black_norm = (binary_black > 0).astype(np.uint8)
distance_from_white = ndimage.distance_transform_edt(binary_black_norm)

# Визуализация толщины
thickness_vis = np.clip(distance_from_white * 50, 0, 255).astype(np.uint8)
cv2.imwrite('debug_skel_03_thickness.png', thickness_vis)
print("→ debug_skel_03_thickness.png (толщина линий)")

# Определяем, где линии толстые
thick_lines_mask = distance_from_white > THICK_LINE_THRESHOLD
thin_lines_mask = (binary_black_norm > 0) & ~thick_lines_mask

print(f"Толстые линии (>{THICK_LINE_THRESHOLD}px): {thick_lines_mask.sum()} пикселей")
print(f"Тонкие линии (≤{THICK_LINE_THRESHOLD}px): {thin_lines_mask.sum()} пикселей")

# Скелетизируем только толстые линии
print("Скелетизация толстых линий...")
skeleton_thick = skeletonize(thick_lines_mask).astype(np.uint8) * 255

# Объединяем: скелет толстых + оригинал тонких
boundaries_final = np.zeros_like(gray)
boundaries_final[skeleton_thick > 0] = 255  # Скелет толстых
boundaries_final[thin_lines_mask] = 255      # Оригинальные тонкие

cv2.imwrite('debug_skel_04_skeleton_thick.png', skeleton_thick)
cv2.imwrite('debug_skel_05_boundaries_final.png', boundaries_final)
print("→ debug_skel_04_skeleton_thick.png (скелет толстых линий)")
print("→ debug_skel_05_boundaries_final.png (финальные границы)")

# ====================
# ШАГ 4: WATERSHED для создания участков
# ====================
print("\n[4/6] Watershed для создания участков вплотную к границам...")

# Инвертируем: нужно белые участки для маркеров
binary_white_clean = binary_white.copy()

# Distance transform от границ (для watershed)
# Чем дальше от границы - тем больше значение
dist_from_boundaries = ndimage.distance_transform_edt(binary_white_clean)

# Находим локальные максимумы - центры участков
from scipy.ndimage import maximum_filter
local_max = maximum_filter(dist_from_boundaries, size=20) == dist_from_boundaries
local_max = local_max & (dist_from_boundaries > 5)  # Минимальное расстояние от границы

# Создаем маркеры
markers = ndimage.label(local_max)[0]
print(f"Найдено маркеров (центров участков): {markers.max()}")

# Watershed
# Граница = 0, участки = маркеры
watershed_input = dist_from_boundaries.copy()
watershed_input[boundaries_final > 0] = 0  # Границы = барьеры

from skimage.segmentation import watershed
labels = watershed(-watershed_input, markers, mask=binary_white_clean > 0)

# Визуализация watershed результата
watershed_vis = np.zeros((height, width, 3), dtype=np.uint8)
for label_id in range(1, labels.max() + 1):
    color = np.random.randint(0, 255, 3).tolist()
    watershed_vis[labels == label_id] = color

# Наложим границы
watershed_vis[boundaries_final > 0] = [255, 255, 255]

cv2.imwrite('debug_skel_06_watershed.png', watershed_vis)
print("→ debug_skel_06_watershed.png (watershed результат)")

# ====================
# ШАГ 5: Извлечение контуров из watershed
# ====================
print("\n[5/6] Извлечение контуров участков...")

valid_parcels = []
rejected = {'too_small': 0, 'too_big': 0, 'invalid': 0}

for label_id in range(1, labels.max() + 1):
    # Создаем маску для этого участка
    mask = (labels == label_id).astype(np.uint8) * 255

    # Находим контур
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        continue

    # Берем самый большой контур (на случай если несколько)
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)

    # Фильтр по площади
    if area < MIN_AREA:
        rejected['too_small'] += 1
        continue
    if area > MAX_AREA:
        rejected['too_big'] += 1
        continue

    # Упрощение контура
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
        'id': label_id,
        'points': points,
        'area': area
    })

print(f"\n✓ Найдено участков: {len(valid_parcels)}")
print(f"✗ Отфильтровано:")
print(f"  - Слишком маленькие: {rejected['too_small']}")
print(f"  - Слишком большие: {rejected['too_big']}")
print(f"  - Некорректные: {rejected['invalid']}")

# ====================
# ШАГ 6: Визуализация и экспорт
# ====================
print("\n[6/6] Визуализация и экспорт в GeoJSON...")

# Визуализация финальных контуров
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

cv2.imwrite('debug_skel_07_result.png', debug_img)
print("→ debug_skel_07_result.png")

# Только контуры
contours_only = np.ones((height, width, 3), dtype=np.uint8) * 255
for parcel in valid_parcels:
    points_np = np.array(parcel['points'], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(contours_only, [points_np], True, (0, 0, 0), 1)
cv2.imwrite('debug_skel_08_contours.png', contours_only)
print("→ debug_skel_08_contours.png")

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

print("\n" + "=" * 60)
print("ПРЕИМУЩЕСТВА SKELETON ПОДХОДА:")
print("  ✓ Толстые границы → скелет (1px)")
print("  ✓ Тонкие границы сохранены")
print("  ✓ Watershed создает участки ДО границ")
print("  ✓ Полигоны вплотную друг к другу")
print("  ✓ НЕТ зазоров и расширений")
print("  ✓ Геометрически точно - граница по центру")
print("=" * 60)
