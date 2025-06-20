import os
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from torchvision import transforms
from tqdm import tqdm

# Class groups
OLD_AUG_CLASSES = ['broken stitch', 'needle mark', 'pinched fabric']
COMBO_ONLY_CLASSES = ['hole', 'horizontal', 'vertical']

INPUT_DIR = 'FabricDefectDataset/train'

# Noise
def add_noise(img, amount=0.02):
    arr = np.array(img).astype(np.float32) / 255.0
    noise = np.random.normal(0, amount, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 1)
    return Image.fromarray((noisy_arr * 255).astype(np.uint8))

# Color jitter
color_jitter = transforms.ColorJitter(
    brightness=0.2,
    contrast=0.2,
    saturation=0.2
)

# Process each class
for cls in OLD_AUG_CLASSES + COMBO_ONLY_CLASSES:
    cls_dir = os.path.join(INPUT_DIR, cls)
    images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for img_name in tqdm(images, desc=f"Augmenting {cls}"):
        img_path = os.path.join(cls_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        base_name = os.path.splitext(img_name)[0]

        # OLD AUGS (flip, rotate, noise)
        if cls in OLD_AUG_CLASSES:
            img_flip = ImageOps.mirror(img)
            img_flip.save(os.path.join(cls_dir, f"{base_name}_flip.jpg"))

            img_rot90 = img.rotate(90)
            img_rot90.save(os.path.join(cls_dir, f"{base_name}_rotate90.jpg"))

            img_rot180 = img.rotate(180)
            img_rot180.save(os.path.join(cls_dir, f"{base_name}_rotate180.jpg"))

            img_noise = add_noise(img)
            img_noise.save(os.path.join(cls_dir, f"{base_name}_noise.jpg"))

        # COMBO AUG (jitter + blur + noise)
        if cls in COMBO_ONLY_CLASSES:
            img_aug = color_jitter(img)
            img_aug = img_aug.filter(ImageFilter.GaussianBlur(radius=1.2))
            img_aug = add_noise(img_aug)
            img_aug.save(os.path.join(cls_dir, f"{base_name}_combo.jpg"))
