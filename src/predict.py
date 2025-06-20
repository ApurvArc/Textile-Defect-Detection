import os
import torch
from torchvision import transforms, models
from PIL import Image

# --- Config ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pt")
IMAGE_DIR = os.path.join(BASE_DIR, "test_images")  # folder containing test images
CLASS_NAMES = [
    'broken stitch', 'defect free', 'hole', 'horizontal',
    'lines', 'needle mark', 'pinched fabric', 'stain', 'vertical'
]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --- Load Model ---
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- Predict Single Image ---
def predict_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)
            pred_class = torch.argmax(output, dim=1).item()
        print(f"🖼️ {os.path.basename(image_path)} → Predicted: {CLASS_NAMES[pred_class]}")
    except Exception as e:
        print(f"❌ Failed to process {image_path}: {e}")

# --- Predict on Folder ---
def predict_all_from_folder(folder_path):
    if not os.path.isdir(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return
    image_files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if not image_files:
        print("❌ No image files found in the folder.")
        return
    for fname in image_files:
        predict_image(os.path.join(folder_path, fname))

# --- Run ---
if __name__ == "__main__":
    predict_all_from_folder(IMAGE_DIR)
