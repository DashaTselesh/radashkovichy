import cv2
import numpy as np
from skimage.morphology import skeletonize

# ====================
# ПАРАМЕТРЫ
# ====================
INPUT_IMAGE = 'map_1891_transparent.png'
OUTPUT_SKELETON = 'map_1891_skeleton_simplified.png'

# Сглаживание скелета
SMOOTH_SKELETON = True  # True = сгладить скелет морфологическими операциями
SMOOTHING_METHOD = 'pruning'  # 'closing', 'pruning', 'median'
PRUNE_ITERATIONS = 2  # Количество итераций pruning (удаление коротких выступов)

# Удаление изолированных точек
REMOVE_ISOLATED_PIXELS = True  # Удалить маленькие связные компоненты (точки, шум)
MIN_COMPONENT_SIZE = 10  # Минимальный размер компонента в пикселях (меньше = удалить)

print("=" * 60)
print("СОЗДАНИЕ УПРОЩЕННОГО СКЕЛЕТА ДЛЯ QGIS POLYGONIZE")
print("=" * 60)
print(f"Входной файл: {INPUT_IMAGE}")
print(f"Выходной файл: {OUTPUT_SKELETON}")
print(f"Сглаживание скелета: {SMOOTH_SKELETON}")
if SMOOTH_SKELETON:
    print(f"Метод сглаживания: {SMOOTHING_METHOD}")
    if SMOOTHING_METHOD == 'pruning':
        print(f"Итераций pruning: {PRUNE_ITERATIONS}")
print(f"Удаление изолированных точек: {REMOVE_ISOLATED_PIXELS}")
if REMOVE_ISOLATED_PIXELS:
    print(f"Минимальный размер компонента: {MIN_COMPONENT_SIZE} пикселей")

# ====================
# ШАГ 1: Загрузка с прозрачностью
# ====================
print("\n[1/4] Загрузка изображения...")
img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(f"Файл {INPUT_IMAGE} не найден!")

height, width = img.shape[:2]
print(f"Размер: {width}x{height} пикселей")

# Обработка альфа-канала
alpha_mask = None
if len(img.shape) == 3 and img.shape[2] == 4:
    print("Обнаружен альфа-канал")
    alpha = img[:, :, 3]
    rgb = img[:, :, :3]
    alpha_mask = alpha > 50
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
else:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

# ====================
# ШАГ 2: Бинаризация и маскирование
# ====================
print("\n[2/4] Бинаризация...")
_, binary_white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Применяем альфа-маску
if alpha_mask is not None:
    binary_white[~alpha_mask] = 0
    print("Применена альфа-маска (фон обнулен)")

# ====================
# ШАГ 3: Скелетизация
# ====================
print("\n[3/4] Скелетизация черных границ...")
binary_black = cv2.bitwise_not(binary_white)

# Применяем маску к черным линиям
if alpha_mask is not None:
    binary_black[~alpha_mask] = 0

# Скелетизация
skeleton_bool = skeletonize(binary_black > 0)
skeleton = (skeleton_bool * 255).astype(np.uint8)

cv2.imwrite('debug_skeleton_01_original.png', skeleton)
print("→ debug_skeleton_01_original.png (оригинальный скелет 1px)")

original_pixels = np.sum(skeleton > 0)
print(f"Пикселей в скелете: {original_pixels}")

# ====================
# ШАГ 3.5: Удаление изолированных точек
# ====================
if REMOVE_ISOLATED_PIXELS:
    print(f"\n[3.5/4] Удаление изолированных точек (мин. размер: {MIN_COMPONENT_SIZE}px)...")

    # Найти все связные компоненты
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton, connectivity=8)

    # Создать новую маску без маленьких компонент
    skeleton_cleaned = np.zeros_like(skeleton)
    removed_count = 0

    for i in range(1, num_labels):  # Пропускаем фон (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= MIN_COMPONENT_SIZE:
            skeleton_cleaned[labels == i] = 255
        else:
            removed_count += 1

    skeleton = skeleton_cleaned

    cleaned_pixels = np.sum(skeleton > 0)
    print(f"  Удалено маленьких компонент: {removed_count}")
    print(f"  Пикселей до: {original_pixels}, после: {cleaned_pixels}")

    cv2.imwrite('debug_skeleton_01b_cleaned.png', skeleton)
    print("  → debug_skeleton_01b_cleaned.png (без изолированных точек)")

# ====================
# ШАГ 4: Сглаживание скелета
# ====================
if SMOOTH_SKELETON:
    print(f"\n[4/4] Сглаживание скелета (метод: {SMOOTHING_METHOD})...")

    if SMOOTHING_METHOD == 'pruning':
        # Метод: Pruning - удаление коротких выступов/пиков
        # Удаляем концевые пиксели (endpoints) итеративно

        print(f"  Pruning: удаление выступов ({PRUNE_ITERATIONS} итераций)...")
        skeleton_pruned = skeleton.copy()

        # Kernel для подсчета соседей (8-connectivity)
        kernel_neighbors = np.ones((3, 3), dtype=np.uint8)

        for iteration in range(PRUNE_ITERATIONS):
            # Найти концевые пиксели (те, у которых только 1 сосед)
            # Используем convolution для подсчета соседей
            skeleton_binary = (skeleton_pruned > 0).astype(np.uint8)

            # Подсчитываем соседей для каждого пикселя
            from scipy import ndimage
            neighbor_count = ndimage.convolve(skeleton_binary, kernel_neighbors, mode='constant', cval=0)

            # Концевые точки: сам пиксель = 1, и ровно 2 соседа (1 сосед + сам пиксель = 2)
            endpoints = (skeleton_binary == 1) & (neighbor_count == 2)

            # Удаляем концевые точки
            skeleton_pruned[endpoints] = 0

            removed = np.sum(endpoints)
            print(f"    Итерация {iteration+1}: удалено {removed} концевых пикселей")

            if removed == 0:
                print(f"    Остановка: нет больше концевых пикселей")
                break

        skeleton = skeleton_pruned
        cv2.imwrite('debug_skeleton_02_pruned.png', skeleton)
        print(f"  → debug_skeleton_02_pruned.png (после pruning)")

    elif SMOOTHING_METHOD == 'median':
        # Метод: Медианный фильтр для сглаживания
        print("  Применение медианного фильтра...")
        skeleton_median = cv2.medianBlur(skeleton, 3)

        # Бинаризация после фильтра
        _, skeleton = cv2.threshold(skeleton_median, 127, 255, cv2.THRESH_BINARY)

        # Скелетизация для возврата к 1px
        skeleton_bool = skeletonize(skeleton > 0)
        skeleton = (skeleton_bool * 255).astype(np.uint8)

        cv2.imwrite('debug_skeleton_02_median.png', skeleton)
        print("  → debug_skeleton_02_median.png (после медианного фильтра)")

    else:  # 'closing'
        # Оригинальный метод: Closing + повторная скелетизация
        print("  Morphological closing...")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skeleton_closed = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel)

        # Повторная скелетизация
        skeleton_bool = skeletonize(skeleton_closed > 0)
        skeleton = (skeleton_bool * 255).astype(np.uint8)

        cv2.imwrite('debug_skeleton_02_closed.png', skeleton)
        print("  → debug_skeleton_02_closed.png (после closing)")

    cv2.imwrite('debug_skeleton_03_smoothed.png', skeleton)
    print(f"  → debug_skeleton_03_smoothed.png (финальный результат)")

    pixels_after = np.sum(skeleton > 0)
    print(f"  Пикселей после сглаживания: {pixels_after}")

# ====================
# Сохранение результата
# ====================
print(f"\nСохранение результата: {OUTPUT_SKELETON}")

# Инвертируем (чтобы линии были черными на белом фоне для QGIS)
output = cv2.bitwise_not(skeleton)

# Применяем маску к выходному изображению
if alpha_mask is not None:
    # Где прозрачность была - делаем БЕЛЫМ (чтобы не сливалось с линиями)
    output[~alpha_mask] = 255

cv2.imwrite(OUTPUT_SKELETON, output)
print(f"→ {OUTPUT_SKELETON}")
print("Формат: черные линии на белом фоне (прозрачность = белый)")

print("\n" + "=" * 60)
print("ГОТОВО!")
print("=" * 60)
print("\nДальнейшие шаги в QGIS:")
print("1. Откройте файл:", OUTPUT_SKELETON)
print("2. Raster → Conversion → Polygonize")
print("3. Укажите Band: 1")
print("4. Отметьте 'Use 8-connectedness'")
print("5. Создайте полигоны")
print("6. Отфильтруйте линии по площади или compactness")
print("=" * 60)
