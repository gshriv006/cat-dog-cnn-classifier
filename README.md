# Cat vs Dog Image Classifier
A deep learning project that classifies images as **Cat** or **Dog** using Convolutional Neural Networks (CNN) built with TensorFlow/Keras, with Transfer Learning via MobileNetV2.

---

## Objective
- Build a CNN that classifies images as Cat or Dog
- Train on augmented image data to improve generalisation
- Evaluate using accuracy, loss curves, precision, recall, and F1-score
- Apply Transfer Learning (MobileNetV2) to significantly boost accuracy on a small dataset
- Extend the pipeline to identify the specific breed of the detected animal

---

## Dataset & Configuration
| Parameter | Value |
|---|---|
| Image Size | 150 × 150 px |
| Batch Size | 32 |
| Epochs | 20 |
| Learning Rate | 0.001 |
| Train Split | 80% |
| Test Split | 20% |

Images are organised into `cats/` and `dogs/` subdirectories under `train/` and `test/`. If the dataset is not pre-split, Cell 3b handles the split automatically.

---

## Data Augmentation
Applied only to training data to prevent overfitting:

| Technique | Value |
|---|---|
| Rotation | ±20° |
| Width/Height Shift | ±20% |
| Shear | 15% |
| Zoom | ±20% |
| Horizontal Flip | Random |
| Rescale | Pixels → [0, 1] |

> Test/Validation data is **only rescaled** — no augmentation applied.

---

## CNN Architecture
