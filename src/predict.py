import os
import csv
import torch
from torchvision import transforms, models
from PIL import Image

# --- Paths (Relative to project root) ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pt")
IMAGE_DIR = os.path.join(BASE_DIR, "FabricDefectDataset", "test")
OUTPUT_CSV = os.path.join(BASE_DIR, "report", "predictions.csv")
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# --- Class Info ---
CLASS_NAMES = [
    'broken stitch', 'defect free', 'hole', 'horizontal',
    'lines', 'needle mark', 'pinched fabric', 'stain', 'vertical'
]
NUM_CLASSES = len(CLASS_NAMES)

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transforms ---
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

# --- Predict One Image ---
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()
    return CLASS_NAMES[pred_class]

# --- Predict All Images in Nested Folder Structure ---
def predict_all_from_folder(folder_path):
    results = []
    for root, _, files in os.walk(folder_path):
        for fname in sorted(files):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(root, fname)
                true_label = os.path.basename(os.path.dirname(image_path))
                pred_label = predict_image(image_path)
                print(f"🖼️ {fname} | Actual: {true_label} → Predicted: {pred_label}")
                results.append([fname, true_label, pred_label])
    return results

# --- Save Predictions to CSV ---
def save_predictions_to_csv(rows, csv_path):
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "actual_label", "predicted_label"])
        writer.writerows(rows)

# --- Main ---
if __name__ == "__main__":
    predictions = predict_all_from_folder(IMAGE_DIR)
    save_predictions_to_csv(predictions, OUTPUT_CSV)
