# 🧵 Fabric Defect Detection using ResNet18

A deep learning-based image classification system designed to detect and classify common defects in textile fabrics. The system uses transfer learning with ResNet18 and achieves **97.38% accuracy** across 9 fabric classes.

---

## 📁 Dataset

The dataset consists of high-resolution textile images categorized into:

- `defect free`
- `broken stitch`
- `hole`
- `horizontal`
- `vertical`
- `lines`
- `needle mark`
- `pinched fabric`
- `stain`

> **Total test images**: 2941  
> **Model accuracy**: 97.38%

---

## 🧠 Model Details

- **Base Model**: ResNet18 (ImageNet weights)
- **Framework**: PyTorch
- **Input Size**: 224 × 224 RGB
- **Loss Function**: Cross Entropy
- **Optimizer**: Adam
- **Augmentations**: Class-balanced (flip, rotate, noise, jitter, mild blur)

---

## 📊 Results (Test Set)

| Class           | Accuracy |
|----------------|----------|
| broken stitch  | 100.00%  |
| defect free    | 95.98%   |
| hole           | 97.16%   |
| horizontal     | 98.53%   |
| lines          | 99.36%   |
| needle mark    | 100.00%  |
| pinched fabric | 100.00%  |
| stain          | 100.00%  |
| vertical       | 97.50%   |

---

## 🧪 Features

- Multi-class fabric defect classification
- Balanced training via targeted augmentations
- Visual reports (confusion matrix, ROC curves, per-class accuracy)
- Save misclassified images for manual inspection
- Easily extendable and modular code

---

## 🗂️ Folder Structure

.
├── FabricDefectDataset/      # Organized dataset (train/val/test)
├── models/                   # Saved trained model (.pt file)
├── report/                   # Evaluation reports and visualizations
├── src/
│   ├── train.py              # Model training script
│   ├── evaluate.py           # Evaluation + confusion matrix, ROC
│   ├── predict.py            # Predict labels for new images/folders
│   ├── process.py            # Augmentation for minority classes
│   ├── split_dataset.py      # Stratified train/val/test split
├── README.md                 # Project description and usage
└── .gitattributes

