import torch
import numpy as np
from PIL import Image, ImageDraw
import glob
import os

print("🎨 Подготовка к созданию GIF...")

# Словарь классов для ДЕТЕКЦИИ (Faster R-CNN COCO)
det_classes = {
    1: "person",
    2: "bicycle",
    3: "car",
    17: "cat",
    18: "dog",
    19: "horse",
    64: "potted plant",
    67: "dining table",
}

# Словарь классов для СЕГМЕНТАЦИИ (DeepLabV3 COCO_WITH_VOC)
seg_classes = {
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor",
}

# Цвета для сегментации (номера классов из seg_classes!)
seg_colors = {
    8: (255, 165, 0),  # cat - оранжевый
    12: (139, 69, 19),  # dog - коричневый
    15: (255, 0, 0),  # person - красный
    7: (0, 255, 0),  # car - зеленый
}

det_files = sorted(glob.glob("frames/det_frame_*.pt"))
seg_files = sorted(glob.glob("frames/seg_frame_*.npy"))

if not det_files:
    print("❌ Ошибка: не найдены файлы det_*.pt!")
    exit()

print(f"📁 Найдено кадров: {len(det_files)}")

# === УМЕНЬШАЕМ РАЗМЕР ДЛЯ GIF (чтобы уложиться в 100 МБ лимит GitHub) ===
MAX_WIDTH = 1280
MAX_HEIGHT = 720

first_frame_path = det_files[0].replace("det_", "").replace(".pt", ".png")
first_img = Image.open(first_frame_path)
original_size = first_img.size

ratio = min(MAX_WIDTH / original_size[0], MAX_HEIGHT / original_size[1])
target_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
print(f"📏 Исходный размер: {original_size}, уменьшаем до: {target_size}")

frames_det_gif = []
frames_seg_gif = []
skipped_frames = 0

for i, det_path in enumerate(det_files):
    base_name = os.path.basename(det_path).replace("det_", "").replace(".pt", "")
    img_path = f"frames/{base_name}.png"
    seg_path = f"frames/seg_{base_name}.npy"

    if not os.path.exists(img_path) or not os.path.exists(seg_path):
        print(f"  ⚠️  Пропущен {base_name} (нет файлов)")
        skipped_frames += 1
        continue

    try:
        # Открываем и уменьшаем изображение
        img = Image.open(img_path).convert("RGB")
        img = img.resize(target_size, Image.Resampling.LANCZOS)

        # Коэффициент масштабирования для bounding box
        scale_x = target_size[0] / original_size[0]
        scale_y = target_size[1] / original_size[1]

        # === ДЕТЕКЦИЯ ===
        img_det = img.copy()
        draw_det = ImageDraw.Draw(img_det)

        det_data = torch.load(det_path, map_location="cpu", weights_only=False)
        boxes = det_data[0]["boxes"].cpu().numpy()
        labels = det_data[0]["labels"].cpu().numpy()
        scores = det_data[0]["scores"].cpu().numpy()

        objects_count = 0
        for box, label, score in zip(boxes, labels, scores):
            if score > 0.5:
                x1, y1, x2, y2 = box

                # Масштабируем координаты
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                # ВАЖНО: сортируем координаты (x1 <= x2, y1 <= y2)
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                # Ограничиваем границами изображения
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(target_size[0], x2)
                y2 = min(target_size[1], y2)

                # Проверяем, что рамка не вырожденная
                if x2 <= x1 or y2 <= y1:
                    continue

                class_name = det_classes.get(int(label), f"class_{label}")
                draw_det.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
                text = f"{class_name}: {score:.2f}"
                draw_det.text((x1, max(0, y1 - 12)), text, fill=(255, 255, 0))
                objects_count += 1

        frames_det_gif.append(img_det)

        # === СЕГМЕНТАЦИЯ ===
        img_seg = img.copy()
        seg_mask = np.load(seg_path)

        # Масштабируем маску до размера изображения
        if seg_mask.shape[0] != target_size[1] or seg_mask.shape[1] != target_size[0]:
            mask_pil = Image.fromarray(seg_mask.astype(np.uint8))
            mask_pil = mask_pil.resize(target_size, Image.Resampling.NEAREST)
            seg_mask = np.array(mask_pil)

        # Создаем цветной оверлей
        overlay_array = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        for class_id, color in seg_colors.items():
            mask_pixels = np.where(seg_mask == class_id)
            if len(mask_pixels[0]) > 0:
                overlay_array[mask_pixels[0], mask_pixels[1]] = color

        overlay = Image.fromarray(overlay_array)
        img_seg = Image.blend(img_seg, overlay, alpha=0.5)
        frames_seg_gif.append(img_seg)

        print(f"  ✅ {base_name}: {objects_count} объектов")

    except Exception as e:
        print(f"  ❌ Ошибка {base_name}: {e}")
        skipped_frames += 1
        continue

# === СОХРАНЕНИЕ GIF ===
print(f"\n Сохраняем GIF...")
print(f"   Детекция: {len(frames_det_gif)} кадров")
print(f"   Сегментация: {len(frames_seg_gif)} кадров")
print(f"   Пропущено: {skipped_frames}")

if len(frames_det_gif) > 0:
    frames_det_gif[0].save(
        "detection.gif",
        save_all=True,
        append_images=frames_det_gif[1:],
        duration=500,
        loop=0,
        optimize=True,
    )
    size_mb = os.path.getsize("detection.gif") / (1024 * 1024)
    print(f"✅ detection.gif сохранен! Размер: {size_mb:.2f} МБ")

if len(frames_seg_gif) > 0:
    frames_seg_gif[0].save(
        "segmentation.gif",
        save_all=True,
        append_images=frames_seg_gif[1:],
        duration=500,
        loop=0,
        optimize=True,
    )
    size_mb = os.path.getsize("segmentation.gif") / (1024 * 1024)
    print(f"✅ segmentation.gif сохранен! Размер: {size_mb:.2f} МБ")

print("\n Задания 5 и 6 выполнены!")
