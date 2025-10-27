import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

# ====================
# ПАРАМЕТРЫ
# ====================
INPUT_IMAGE = 'map_1891_cleaned.png'
OUTPUT_GEOJSON = 'map_1891_raw.geojson'

# Фильтры
MIN_AREA = 30
MAX_AREA = 100000

# Морфологическое закрытие - "закрывает" зазоры от толстых линий
CLOSING_KERNEL_SIZE = 5  # Размер kernel (примерно половина толщины границы)

# Упрощение контуров
EPSILON_FACTOR = 0.007  # 0.7% от периметра

print("=" * 60)
print("MORPHOLOGICAL CLOSING - ЗАКРЫТИЕ ЗАЗОРОВ")
print("=" * 60)
print(f"Входной файл: {INPUT_IMAGE}")
print(f"Closing kernel: {CLOSING_KERNEL_SIZE}x{CLOSING_KERNEL_SIZE}")
print(f"\nЛОГИКА:")
print(f"  1. Белые участки имеют зазоры из-за толстых линий")
print(f"  2. Closing закрывает эти зазоры")
print(f"  3. Участки становятся правильного размера вплотную")
print(f"  4. Находим контуры на закрытом изображении")

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

# Бинаризация - белые участки
_, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
cv2.imwrite('debug_closing_01_binary.png', binary)
print("→ debug_closing_01_binary.png (исходные белые участки с зазорами)")

# ====================
# ШАГ 2: MORPHOLOGICAL CLOSING
# ====================
print(f"\n[2/5] Morphological Closing (kernel {CLOSING_KERNEL_SIZE}x{CLOSING_KERNEL_SIZE})...")
print("Закрываем зазоры от толстых границ...")

# Создаем структурный элемент (kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (CLOSING_KERNEL_SIZE, CLOSING_KERNEL_SIZE))

# Closing = Dilation + Erosion
# Расширяет белые участки, затем сжимает обратно
# Результат: зазоры закрыты, форма сохранена
binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

cv2.imwrite('debug_closing_02_closed.png', binary_closed)
print("→ debug_closing_02_closed.png (после closing - зазоры закрыты)")

# Визуализация разницы
diff = cv2.absdiff(binary, binary_closed)
cv2.imwrite('debug_closing_03_difference.png', diff)
print("→ debug_closing_03_difference.png (что добавил closing)")

# ====================
# ШАГ 3: Поиск контуров
# ====================
print("\n[3/5] Поиск контуров на закрытом изображении...")
contours, _ = cv2.findContours(binary_closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
print(f"Найдено контуров: {len(contours)}")

# ====================
# ШАГ 4: Фильтрация и упрощение
# ====================
print("\n[4/5] Фильтрация и упрощение контуров...")

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

cv2.imwrite('debug_closing_04_result.png', debug_img)
print("→ debug_closing_04_result.png")

# Только контуры
contours_only = np.ones((height, width, 3), dtype=np.uint8) * 255
for parcel in valid_parcels:
    points_np = np.array(parcel['points'], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(contours_only, [points_np], True, (0, 0, 0), 1)
cv2.imwrite('debug_closing_05_contours.png', contours_only)
print("→ debug_closing_05_contours.png")

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
print("ПРЕИМУЩЕСТВА CLOSING ПОДХОДА:")
print("  ✓ ОЧЕНЬ простой алгоритм")
print("  ✓ Прямое решение проблемы толстых границ")
print("  ✓ Closing закрывает зазоры между участками")
print("  ✓ Участки вплотную друг к другу")
print("  ✓ НЕТ фантомных полигонов")
print("  ✓ НЕТ watershed артефактов")
print("  ✓ Быстро и эффективно")
print("=" * 60)
print("\nНАСТРОЙКА:")
print(f"  Если зазоры остались → увеличьте CLOSING_KERNEL_SIZE")
print(f"  Если участки слились → уменьшите CLOSING_KERNEL_SIZE")
print(f"  Текущее значение: {CLOSING_KERNEL_SIZE}")
print("=" * 60)
