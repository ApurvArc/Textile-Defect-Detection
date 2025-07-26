import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize

# --- Config ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "FabricDefectDataset", "test")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "report")

BATCH_SIZE = 32
NUM_CLASSES = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --- Dataset & Loader ---
test_set = datasets.ImageFolder(DATA_DIR, transform=transform)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
class_names = test_set.classes

# --- Save class names ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "class_names.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(class_names))

# --- Load model ---
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- Evaluate on test set ---
y_true = []
y_pred = []
y_score = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_score.extend(outputs.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_score = np.array(y_score)

# --- Metrics ---
acc = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
cm = confusion_matrix(y_true, y_pred)

# --- Save report (txt and csv) ---
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write("Fabric Defect Classification Report (Test Set)\n")
    f.write("=" * 45 + "\n")
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write(classification_report(y_true, y_pred, target_names=class_names, digits=4))

pd.DataFrame(report).transpose().to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"))

# --- Confusion Matrix (Raw) ---
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ConfMatrix.png"))
plt.close()

# --- Confusion Matrix (Normalized) ---
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(9, 7))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=class_names, yticklabels=class_names)
plt.title("Normalized Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ConfMatrix_Normalized.png"))
plt.close()

# --- Per-Class Accuracy ---
class_acc = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(9, 6))
sns.barplot(x=class_names, y=class_acc)
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Per-Class Accuracy (Test Set)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "PerClassAccuracy.png"))
plt.close()

# --- ROC Curve (One-vs-All) ---
y_true_bin = label_binarize(y_true, classes=np.arange(NUM_CLASSES))
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(NUM_CLASSES):
    plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (Test Set)")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ROC_Curves.png"))
plt.close()
