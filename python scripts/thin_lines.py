import cv2
import numpy as np

# ====================
# ПАРАМЕТРЫ
# ====================
INPUT_IMAGE = 'map_1891.png'
OUTPUT_IMAGE = 'map_1891_thinned.png'

print("=" * 60)
print("УТОНЧЕНИЕ ЛИНИЙ ДО 1 ПИКСЕЛЯ")
print("=" * 60)

# ====================
# ШАГ 1: Загрузка
# ====================
print("\n[1/5] Загрузка изображения...")
img = cv2.imread(INPUT_IMAGE)
if img is None:
    raise FileNotFoundError(f"Файл {INPUT_IMAGE} не найден!")

height, width = img.shape[:2]
print(f"Размер: {width}x{height} пикселей")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ====================
# ШАГ 2: Бинаризация
# ====================
print("\n[2/5] Адаптивная бинаризация...")
binary = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,  # Черные линии становятся белыми
    blockSize=15,
    C=5
)

cv2.imwrite('thin_debug_01_binary.png', binary)
print("→ Сохранено: thin_debug_01_binary.png")

# ====================
# ШАГ 3: Замыкание разрывов (очень мягкое)
# ====================
print("\n[3/5] Замыкание разрывов в линиях...")
kernel_close = np.ones((3, 3), np.uint8)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)

cv2.imwrite('thin_debug_02_closed.png', closed)
print("→ Сохранено: thin_debug_02_closed.png")

# ====================
# ШАГ 4: Эрозия для утончения линий
# ====================
print("\n[4/5] Эрозия для утончения линий...")

# 3 итерации эрозии с маленьким ядром для более тонких линий
kernel_erode = np.ones((2, 2), np.uint8)
thinned = cv2.erode(closed, kernel_erode, iterations=1)

cv2.imwrite('thin_debug_03_eroded.png', thinned)
print("→ Сохранено: thin_debug_03_eroded.png")

# Пропускаем MORPH_OPEN - он тоже удаляет линии
cleaned = thinned

# ====================
# ШАГ 5: Инверсия обратно (белый фон, черные линии)
# ====================
print("\n[5/5] Инверсия обратно (белый фон, черные линии)...")
result = cv2.bitwise_not(cleaned)

cv2.imwrite(OUTPUT_IMAGE, result)
print(f"→ Сохранено: {OUTPUT_IMAGE}")

# ====================
# Статистика
# ====================
print("\n" + "=" * 60)
print("СТАТИСТИКА:")
print("=" * 60)

# Подсчет пикселей линий
original_line_pixels = np.sum(binary > 0)
thinned_line_pixels = np.sum(cleaned > 0)

print(f"Пиксели линий ДО утончения: {original_line_pixels:,}")
print(f"Пиксели линий ПОСЛЕ утончения: {thinned_line_pixels:,}")
print(f"Уменьшение: {((original_line_pixels - thinned_line_pixels) / original_line_pixels * 100):.1f}%")

print("=" * 60)
print("\n✓ ГОТОВО! Проверьте файлы:")
print("  1. thin_debug_01_binary.png - бинаризация")
print("  2. thin_debug_02_closed.png - замкнутые разрывы")
print("  3. thin_debug_03_eroded.png - утонченные линии")
print(f"  4. {OUTPUT_IMAGE} - РЕЗУЛЬТАТ (черные линии на белом)")
print("\nЭрозия (3 итерации) - линии стали значительно тоньше!")
print("Прямые углы сохранены. Если линии слишком тонкие/разорваны - уменьшите iterations в коде.")
print("=" * 60)
