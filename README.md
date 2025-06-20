# 🧵 Fabric Defect Detection using ResNet18

A deep learning-based image classification system designed to detect and classify common defects in textile fabrics. This project uses transfer learning with ResNet18 and achieves **97.38% accuracy** across 9 fabric classes.

---

## 📂 Dataset

This project uses a **combined dataset** built from two Kaggle sources:

1. **[Fabric Defects Dataset](https://www.kaggle.com/datasets/nexuswho/fabric-defects-dataset)**  
   Contributed by `nexuswho`, this dataset provides:
   - `defect free`, `hole`, `horizontal`, `lines`, `stain`, `vertical`

2. **[Multi-Class Fabric Defect Detection Dataset](https://www.kaggle.com/datasets/ziya07/multi-class-fabric-defect-detection-dataset)**  
   Contributed by `ziya07`, this adds:
   - `broken stitch`, `needle mark`, `pinched fabric`

> **Total test images**: 2941  
> **Model accuracy**: 97.38%

---

## 🧠 Model Details

- **Base Model**: ResNet18 (ImageNet pre-trained)
- **Framework**: PyTorch
- **Input Size**: 224 × 224 RGB
- **Loss Function**: Cross Entropy
- **Optimizer**: Adam
- **Augmentations**: Rotation, noise, color jitter, mild blur (class-balanced)

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

- Multi-class textile defect detection
- Balanced training with targeted augmentation
- Confusion matrix, ROC, per-class accuracy
- Save misclassified images
- Modular PyTorch codebase

---

<pre> ```text ## 📁 Folder Structure . ├── FabricDefectDataset/ # Structured dataset (train/val/test) │ ├── raw/ # Raw unorganized images │ ├── train/ │ ├── val/ │ └── test/ ├── models/ # Trained model weights (.pt) ├── report/ # Evaluation reports & plots ├── src/ # All source code │ ├── train.py # Model training │ ├── evaluate.py # Evaluation & reporting │ ├── predict.py # Inference on new images │ ├── process.py # Data augmentation │ ├── rename.py # Rename images class-wise │ └── split_dataset.py # Stratified split (train/val/test) ├── requirements.txt # Python dependencies ├── README.md # Project documentation └── .gitattributes ``` </pre>