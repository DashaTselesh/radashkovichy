import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

# ====================
# ПАРАМЕТРЫ
# ====================
INPUT_IMAGE = 'map_1891_transparent.png'
OUTPUT_GEOJSON = 'map_1891_raw.geojson'

# Фильтры
MIN_AREA = 200  # Минимальная площадь участка (отфильтрует белые точки-шумы)
MAX_AREA = 2000000  # Максимальная площадь (2 миллиона пикселей)

# Упрощение контуров
EPSILON_FACTOR = 0.02  # 2% от периметра (агрессивное упрощение для прямых линий)

# Опции
SKELETONIZE_BOUNDARIES = True  # True = утончить черные границы до 1px
SKELETON_THICKNESS = 1  # Толщина скелета в пикселях (1 = минимум)
SMOOTH_BEFORE_THRESHOLD = False  # False = НЕ использовать blur, сохранить острые углы!
BLUR_KERNEL_SIZE = 3  # Размер kernel для Gaussian blur (если включено)

print("=" * 60)
print("ИЗВЛЕЧЕНИЕ УЧАСТКОВ ИЗ TRANSPARENT PNG")
print("=" * 60)
print(f"Входной файл: {INPUT_IMAGE}")
print(f"Минимальная площадь: {MIN_AREA}px (фильтр белых точек)")
print(f"Сглаживание ДО бинаризации: {SMOOTH_BEFORE_THRESHOLD}" + (f" (blur {BLUR_KERNEL_SIZE}x{BLUR_KERNEL_SIZE})" if SMOOTH_BEFORE_THRESHOLD else ""))
print(f"Скелетизация границ: {SKELETONIZE_BOUNDARIES}")
print(f"Упрощение контуров: {EPSILON_FACTOR*100:.1f}% (меньше вершин)")

# ====================
# ШАГ 1: Загрузка с поддержкой прозрачности
# ====================
print("\n[1/4] Загрузка изображения с прозрачностью...")
img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(f"Файл {INPUT_IMAGE} не найден!")

height, width = img.shape[:2]
print(f"Размер: {width}x{height} пикселей")
print(f"Каналов: {img.shape[2] if len(img.shape) == 3 else 1}")

# Если есть альфа-канал (RGBA), используем его как маску
alpha_mask = None
if len(img.shape) == 3 and img.shape[2] == 4:
    print("Обнаружен альфа-канал (прозрачность)")
    alpha = img[:, :, 3]
    rgb = img[:, :, :3]

    # Маска: где alpha > 0 = рабочая область, alpha == 0 = игнорируем
    alpha_mask = alpha > 50

    # Берем RGB как есть, БЕЗ смешивания с фоном!
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    print(f"Используем альфа-канал как маску (прозрачность игнорируется)")
    cv2.imwrite('debug_trans_00_alpha_mask.png', (alpha_mask * 255).astype(np.uint8))
    print("→ debug_trans_00_alpha_mask.png (белое=рабочая область, черное=игнор)")
else:
    print("Альфа-канал не найден, используем как есть")
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    alpha_mask = None

cv2.imwrite('debug_trans_01_grayscale.png', gray)
print("→ debug_trans_01_grayscale.png")

# ====================
# ШАГ 2: Сглаживание ДО бинаризации (опционально)
# ====================
if SMOOTH_BEFORE_THRESHOLD:
    print(f"\n[2a/4] Сглаживание ДО бинаризации (Gaussian blur {BLUR_KERNEL_SIZE}x{BLUR_KERNEL_SIZE})...")
    # Gaussian blur сглаживает границы НЕ заполняя пространство
    gray_blurred = cv2.GaussianBlur(gray, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)
    cv2.imwrite('debug_trans_01a_blurred.png', gray_blurred)
    print("→ debug_trans_01a_blurred.png (сглаженные границы)")
    gray = gray_blurred

# ====================
# ШАГ 2: Бинаризация
# ====================
print("\n[2/4] Бинаризация (белые участки vs черные границы)...")

# Threshold - белые участки
_, binary_white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
cv2.imwrite('debug_trans_02a_binary_raw.png', binary_white)
print("→ debug_trans_02a_binary_raw.png (сырая бинаризация)")

# Применяем альфа-маску СРАЗУ к бинаризации
# Это удалит случайные белые точки на прозрачном фоне ДО скелетизации
if alpha_mask is not None:
    binary_white[~alpha_mask] = 0
    cv2.imwrite('debug_trans_02a_binary_masked.png', binary_white)
    print("→ debug_trans_02a_binary_masked.png (белые точки на фоне удалены)")

# Опционально: скелетизировать черные границы
if SKELETONIZE_BOUNDARIES:
    print("\n  Скелетизация черных границ для утончения...")
    from skimage.morphology import skeletonize

    # Инвертируем для скелетизации черных линий (белые границы на черном фоне)
    binary_black = cv2.bitwise_not(binary_white)

    # КЛЮЧЕВОЙ МОМЕНТ 1: Применяем маску ДО скелетизации
    # Обнуляем фон (прозрачные области) чтобы не участвовали в скелетизации
    if alpha_mask is not None:
        binary_black[~alpha_mask] = 0
        print("  → Альфа-маска применена ДО скелетизации (фон обнулен)")

    cv2.imwrite('debug_trans_02b_black_masked.png', binary_black)
    print("  → debug_trans_02b_black_masked.png (черные границы, фон обнулен)")

    # Скелетизируем (границы уже сглажены на шаге 2a)
    skeleton_bool = skeletonize(binary_black > 0)
    skeleton = (skeleton_bool * 255).astype(np.uint8)

    cv2.imwrite('debug_trans_02c_skeleton_1px.png', skeleton)
    print("  → debug_trans_02c_skeleton_1px.png (скелет 1px)")

    # Инвертируем обратно - белые участки расширены до середины границ
    binary_white = cv2.bitwise_not(skeleton)
    cv2.imwrite('debug_trans_02d_white_expanded.png', binary_white)
    print("  → debug_trans_02d_white_expanded.png (белые участки, границы = скелет)")

# КЛЮЧЕВОЙ МОМЕНТ 2: Применяем маску ПОСЛЕ скелетизации/обработки
# Обнуляем фон чтобы findContours не находил его как контур
if alpha_mask is not None:
    binary_white[~alpha_mask] = 0
    print("  → Альфа-маска применена ПОСЛЕ обработки (фон обнулен)")

cv2.imwrite('debug_trans_02e_final_masked.png', binary_white)
print("→ debug_trans_02e_final_masked.png (финал: участки, фон обнулен)")

# ====================
# ШАГ 3: Поиск контуров белых участков
# ====================
print("\n[3/4] Поиск контуров белых участков...")

contours, hierarchy = cv2.findContours(binary_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Найдено контуров: {len(contours)}")

# DEBUG: проверяем что в binary_white
white_pixels = np.sum(binary_white > 0)
black_pixels = np.sum(binary_white == 0)
print(f"DEBUG: Белых пикселей: {white_pixels}, Черных: {black_pixels}")

# Визуализация всех контуров
all_contours_vis = cv2.cvtColor(binary_white, cv2.COLOR_GRAY2BGR)
cv2.drawContours(all_contours_vis, contours, -1, (0, 255, 0), 1)
cv2.imwrite('debug_trans_03_all_contours.png', all_contours_vis)
print("→ debug_trans_03_all_contours.png (все контуры)")

# DEBUG: площади ВСЕХ контуров
if len(contours) > 0:
    areas_all = sorted([cv2.contourArea(c) for c in contours], reverse=True)
    areas_debug = areas_all[:10]
    print(f"Площади топ-10 контуров: {[int(a) for a in areas_debug]}")
    print(f"Всего контуров: {len(contours)}, самый большой: {int(areas_all[0])}, самый маленький: {int(areas_all[-1])}")
else:
    print("WARNING: Контуры НЕ найдены!")

# ====================
# ШАГ 4: Фильтрация и упрощение
# ====================
print("\n[4/4] Фильтрация и упрощение...")

valid_parcels = []
rejected = {'too_small': 0, 'too_big': 0, 'invalid': 0}

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)

    # DEBUG для первых 5 контуров
    if i < 5:
        print(f"  Контур {i}: площадь={int(area)}, MIN={MIN_AREA}, MAX={MAX_AREA}", end="")

    # Фильтр по площади
    if area < MIN_AREA:
        rejected['too_small'] += 1
        if i < 5:
            print(" → too_small (белый шум)")
        continue
    if area > MAX_AREA:
        rejected['too_big'] += 1
        if i < 5:
            print(" → too_big")
        continue

    if i < 5:
        print(" → OK")

    # Упрощение контура участка
    perimeter = cv2.arcLength(contour, True)
    epsilon = EPSILON_FACTOR * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if i < 5:
        vertices_before = len(contour)
        vertices_after = len(approx)
        reduction = (1 - vertices_after/vertices_before) * 100 if vertices_before > 0 else 0
        print(f"    Упрощение контура {i}: {vertices_before} → {vertices_after} вершин ({reduction:.1f}% меньше)")

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
print(f"  - Слишком маленькие (шумы): {rejected['too_small']}")
print(f"  - Слишком большие: {rejected['too_big']}")
print(f"  - Некорректные: {rejected['invalid']}")

# ====================
# Визуализация
# ====================
print("\nВизуализация...")

# Визуализация валидных участков
debug_img = cv2.cvtColor(binary_white, cv2.COLOR_GRAY2BGR)
for parcel in valid_parcels:
    points_np = np.array(parcel['points'], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(debug_img, [points_np], True, (0, 255, 0), 2)

    M = cv2.moments(points_np)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(debug_img, str(parcel['id']), (cx-10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

cv2.imwrite('debug_trans_04_result.png', debug_img)
print("→ debug_trans_04_result.png (валидные участки)")

# Только контуры
contours_only = np.ones((height, width, 3), dtype=np.uint8) * 255
for parcel in valid_parcels:
    points_np = np.array(parcel['points'], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(contours_only, [points_np], True, (0, 0, 0), 1)
cv2.imwrite('debug_trans_05_contours.png', contours_only)
print("→ debug_trans_05_contours.png")

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

if len(valid_parcels) > 0:
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
print("ПРЕИМУЩЕСТВА TRANSPARENT PNG:")
print("  ✓ Фон уже удален вручную")
print("  ✓ Белые участки четко определены")
print("  ✓ Простой поиск контуров белых областей")
print("  ✓ MIN_AREA фильтрует белые точки-шумы")
print("  ✓ Быстро и точно!")
print("=" * 60)
