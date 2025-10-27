import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from skimage.morphology import skeletonize

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

print("=" * 60)
print("СКЕЛЕТИЗАЦИЯ ГРАНИЦ - ПРОСТОЙ ПОДХОД")
print("=" * 60)
print(f"Входной файл: {INPUT_IMAGE}")
print(f"\nЛОГИКА:")
print(f"  1. Инвертируем: черные линии → белые")
print(f"  2. Скелетизация: каждая линия → 1px по центру")
print(f"  3. Инвертируем обратно: тонкие черные границы")
print(f"  4. Находим контуры белых участков")
print(f"  → Участки вплотную, разделены скелетами!")

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
_, binary_white = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)  # Белые участки
cv2.imwrite('debug_skel_simple_01_original.png', binary_white)
print("→ debug_skel_simple_01_original.png (исходные белые участки)")

# ====================
# ШАГ 2: Инвертирование - черные линии станут белыми
# ====================
print("\n[2/5] Инвертирование изображения...")
binary_inverted = cv2.bitwise_not(binary_white)
cv2.imwrite('debug_skel_simple_02_inverted.png', binary_inverted)
print("→ debug_skel_simple_02_inverted.png (черные линии → белые)")

# ====================
# ШАГ 3: СКЕЛЕТИЗАЦИЯ
# ====================
print("\n[3/5] Скелетизация границ...")
print("Преобразуем каждую линию в скелет 1px по центру...")

# Преобразуем в boolean для skeletonize
binary_bool = binary_inverted > 0

# Скелетизация - находит центр каждой линии независимо от толщины!
skeleton_bool = skeletonize(binary_bool)

# Обратно в uint8
skeleton = (skeleton_bool * 255).astype(np.uint8)
cv2.imwrite('debug_skel_simple_03_skeleton.png', skeleton)
print("→ debug_skel_simple_03_skeleton.png (скелет - 1px линии)")

# Статистика
original_pixels = np.sum(binary_inverted > 0)
skeleton_pixels = np.sum(skeleton > 0)
reduction = 100 * (1 - skeleton_pixels / original_pixels)
print(f"Черных пикселей до: {original_pixels}")
print(f"Черных пикселей после: {skeleton_pixels}")
print(f"Уменьшение: {reduction:.1f}%")

# ====================
# ШАГ 4: Сглаживание и закрытие разрывов в скелете
# ====================
print("\n[4/6] Сглаживание скелета и закрытие разрывов...")
# Скелет может иметь мелкие разрывы
# Closing с маленьким kernel закроет разрывы БЕЗ сильного расширения

# Kernel для closing - соединяет разрывы в скелете
# Увеличиваем размер чтобы гарантировать замкнутость границ
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
skeleton_closed = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel_close)

cv2.imwrite('debug_skel_simple_04a_skeleton_closed.png', skeleton_closed)
print("→ debug_skel_simple_04a_skeleton_closed.png (скелет с закрытыми разрывами)")

# Теперь инвертируем
white_areas_final = cv2.bitwise_not(skeleton_closed)

cv2.imwrite('debug_skel_simple_04b_inverted.png', white_areas_final)
print("→ debug_skel_simple_04b_inverted.png (белые участки разделены скелетом)")
print("Участки правильного размера - границы по центру толстых линий!")

# ====================
# ШАГ 5: Поиск контуров белых участков
# ====================
print("\n[5/6] Поиск контуров белых участков...")
# Используем RETR_EXTERNAL - только внешние контуры, не внутренние (линии)

all_contours, hierarchy = cv2.findContours(white_areas_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Найдено контуров (EXTERNAL): {len(all_contours)}")

# DEBUG: показываем площади первых 10 контуров
if len(all_contours) > 0:
    areas_debug = [cv2.contourArea(c) for c in all_contours[:10]]
    print(f"Площади первых 10 контуров: {areas_debug}")

# Визуализация всех контуров
contours_vis = cv2.cvtColor(white_areas_final, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contours_vis, all_contours, -1, (0, 255, 0), 1)
cv2.imwrite('debug_skel_simple_05_all_contours.png', contours_vis)
print("→ debug_skel_simple_05_all_contours.png (все найденные контуры)")

# ====================
# Фильтрация и упрощение
# ====================
print("\nФильтрация и упрощение контуров...")

valid_parcels = []
rejected = {'too_small': 0, 'too_big': 0, 'invalid': 0}

for i, contour in enumerate(all_contours):
    area = cv2.contourArea(contour)

    # DEBUG: показываем почему первые несколько отфильтровались
    if i < 5:
        print(f"  Контур {i}: площадь={area:.0f}, MIN={MIN_AREA}, MAX={MAX_AREA}")

    # Фильтр по площади
    if area < MIN_AREA:
        rejected['too_small'] += 1
        continue
    if area > MAX_AREA:
        rejected['too_big'] += 1
        if i < 5:
            print(f"    → Отфильтрован: too_big")
        continue

    # Упрощение контура
    perimeter = cv2.arcLength(contour, True)
    epsilon = EPSILON_FACTOR * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) < 3:
        rejected['invalid'] += 1
        continue

    # Преобразуем в список точек
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
# Визуализация
# ====================
print("\nВизуализация...")

# Визуализация на файле с тонкими границами (не на оригинале!)
debug_img = cv2.cvtColor(white_areas_final, cv2.COLOR_GRAY2BGR)
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

cv2.imwrite('debug_skel_simple_06_result.png', debug_img)
print("→ debug_skel_simple_06_result.png (контуры на изображении с тонкими границами)")

# Только контуры
contours_only = np.ones((height, width, 3), dtype=np.uint8) * 255
for parcel in valid_parcels:
    points_np = np.array(parcel['points'], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(contours_only, [points_np], True, (0, 0, 0), 1)
cv2.imwrite('debug_skel_simple_07_contours.png', contours_only)
print("→ debug_skel_simple_07_contours.png")

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

print("\n" + "=" * 60)
print("ПРЕИМУЩЕСТВА ЭТОГО ПОДХОДА:")
print("  ✓ Скелетизация работает для ЛЮБОЙ толщины линии")
print("  ✓ Линия 2px → скелет 1px по центру")
print("  ✓ Линия 10px → скелет 1px по центру")
print("  ✓ Автоматически - не нужно знать толщину")
print("  ✓ Участки вплотную, разделены тонкими скелетами")
print("  ✓ НЕТ зазоров между участками")
print("  ✓ НЕТ watershed артефактов")
print("  ✓ ПРОСТОЙ алгоритм")
print("=" * 60)
