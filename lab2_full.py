import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from PIL import Image
import os
import glob
import numpy as np  # <-- ДОБАВИЛИ ЭТО

print("🚀 Загрузка моделей...")

det_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
)
det_model.eval()

seg_model = torchvision.models.segmentation.deeplabv3_resnet50(
    weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
)
seg_model.eval()

print("✅ Модели загружены!")

coco_classes = {
    1: "person",
    2: "bicycle",
    3: "car",
    17: "cat",
    18: "dog",
    19: "horse",
    64: "potted plant",
    67: "dining table",
}

transform = transforms.ToTensor()
frame_files = sorted(glob.glob("frames/frame_*.png"))
print(f"📁 Найдено кадров: {len(frame_files)}")

print("\n🔄 Обрабатываем все кадры...")

for idx, frame_path in enumerate(frame_files):
    frame_name = os.path.basename(frame_path).replace(".png", "")
    print(f"  [{idx+1}/{len(frame_files)}] {frame_name}...")

    img = Image.open(frame_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        det_pred = det_model(img_tensor)
        seg_pred = seg_model(img_tensor)

    # 1. Сохраняем детекцию (она и так весит мало, оставляем как есть)
    torch.save(det_pred, f"frames/det_{frame_name}.pt")

    # 2. Сохраняем сегментацию ПРАВИЛЬНО (только итоговые классы, без тяжелых float)
    # seg_pred['out'] имеет форму [1, 21, Height, Width]
    # argmax(dim=1) выбирает класс с максимальной вероятностью для каждого пикселя -> [1, Height, Width]
    # squeeze(0) убирает размерность батча -> [Height, Width]
    # numpy().astype(np.uint8) превращает в лёгкие целые числа (0-255)
    seg_mask = seg_pred["out"].argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Сохраняем как .npy (это стандартный и очень компактный формат для массивов)
    np.save(f"frames/seg_{frame_name}.npy", seg_mask)

    # Выводим найденные объекты для контроля
    labels = det_pred[0]["labels"].cpu().numpy()
    scores = det_pred[0]["scores"].cpu().numpy()
    found = [
        (coco_classes.get(int(l), f"class_{l}"), s)
        for l, s in zip(labels, scores)
        if s > 0.5
    ]

    if found:
        for name, score in found:
            print(f"    ✓ {name}: {score:.2f}")

print("\n🎉 Задание 4 выполнено!")
