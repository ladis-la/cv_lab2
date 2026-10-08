import numpy as np
import os

# Проверяем первую маску
mask_path = "frames/seg_frame_0001.npy"

if not os.path.exists(mask_path):
    print("❌ Файл маски не найден!")
    exit()

mask = np.load(mask_path)
print(f"📊 Размер маски: {mask.shape}")
print(f"📊 Тип данных: {mask.dtype}")
print(f" Минимальное значение: {mask.min()}")
print(f"📊 Максимальное значение: {mask.max()}")

# Считаем, сколько пикселей каждого класса
unique, counts = np.unique(mask, return_counts=True)
print("\n📋 Распределение классов в маске:")
for class_id, count in zip(unique, counts):
    percentage = (count / mask.size) * 100
    print(f"  Класс {class_id}: {count} пикселей ({percentage:.2f}%)")