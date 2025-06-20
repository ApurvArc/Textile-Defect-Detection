import os
import shutil
import random

# --- Setup Relative Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..", "FabricDefectDataset")
CLASSES = [
    'broken stitch', 'defect free', 'hole', 'horizontal',
    'lines', 'needle mark', 'pinched fabric', 'stain', 'vertical'
]
SPLITS = ['train', 'val', 'test']
SPLIT_RATIOS = (0.7, 0.15, 0.15)  # train, val, test

random.seed(42)

for cls in CLASSES:
    src_dir = os.path.join(BASE_DIR, cls)
    if not os.path.isdir(src_dir):
        print(f"❌ Class folder not found: {src_dir}")
        continue

    files = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    random.shuffle(files)

    total = len(files)
    train_end = int(SPLIT_RATIOS[0] * total)
    val_end = train_end + int(SPLIT_RATIOS[1] * total)

    split_files = {
        'train': files[:train_end],
        'val': files[train_end:val_end],
        'test': files[val_end:]
    }

    for split in SPLITS:
        dest_dir = os.path.join(BASE_DIR, split, cls)
        os.makedirs(dest_dir, exist_ok=True)
        for f in split_files[split]:
            shutil.copy(os.path.join(src_dir, f), os.path.join(dest_dir, f))

print("✅ Dataset split successfully into train/val/test folders.")
