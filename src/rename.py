import os
from PIL import Image

# Dataset root directory
BASE_DIR = r"C:\Users\ARCREACTOR19\Downloads\your_project\FabricDefectDataset"

# Supported image extensions
EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Traverse all class folders
for class_name in os.listdir(BASE_DIR):
    class_path = os.path.join(BASE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path) if f.lower().endswith(EXTS)]
    images.sort()

    print(f"🔄 Renaming {len(images)} images in '{class_name}'...")

    for i, fname in enumerate(images, 1):
        old_path = os.path.join(class_path, fname)
        new_name = f"{class_name.replace(' ', '_')}_{i:03d}.jpg"
        new_path = os.path.join(class_path, new_name)

        try:
            img = Image.open(old_path).convert("RGB")
            img.save(new_path, "JPEG")
            # os.remove(old_path)  # 🔒 Disabled to preserve original
        except Exception as e:
            print(f"❌ Failed to process {fname}: {e}")

print("✅ All images renamed successfully.")
