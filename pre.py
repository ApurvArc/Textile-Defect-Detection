import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import torchvision.transforms as T
import numpy as np
import random

# Load a sample image (adjust path as needed)
image_path = "FabricDefectDataset/train/needle mark/needle_mark_001.jpg"
img = Image.open(image_path).convert("RGB").resize((224, 224))

# Define Gaussian noise function
def add_noise(img, amount=0.02):
    arr = np.array(img).astype(np.float32) / 255.0
    noise = np.random.normal(0, amount, arr.shape)
    noisy = np.clip(arr + noise, 0, 1)
    return Image.fromarray((noisy * 255).astype(np.uint8))

# Define all augmentations
augmentations = [
    ("Original", img),
    ("Horizontal Flip", img.transpose(Image.FLIP_LEFT_RIGHT)),
    ("Rotate 90°", img.rotate(90)),
    ("Rotate 180°", img.rotate(180)),
    ("Gaussian Noise", add_noise(img)),
    ("Color Jitter", T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)(img)),
    ("Blur", img.filter(ImageFilter.GaussianBlur(radius=1.5))),
    ("Warp (Affine)", T.RandomAffine(degrees=0, shear=15)(img)),
    ("Random Crop", T.RandomResizedCrop(224, scale=(0.8, 1.0))(img))
]

# Plot 3x3 grid
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for ax, (title, aug_img) in zip(axes.flat, augmentations):
    ax.imshow(aug_img)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.show()
