# Cat vs Dog Image Classifier
A deep learning project that classifies images as **Cat** or **Dog** using Convolutional Neural Networks (CNN) built with TensorFlow/Keras, with Transfer Learning via MobileNetV2.
---
## Objective
- Build a CNN that classifies images as Cat or Dog
- Train on augmented image data to improve generalisation
- Evaluate using accuracy, loss curves, precision, recall, and F1-score
- Apply Transfer Learning (MobileNetV2) to significantly boost accuracy on a small dataset
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
Images are organised into cats/ and dogs/ subdirectories under train/ and test/. If the dataset is not pre-split, Cell 3b handles the split automatically.
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

Input (150×150×3)
    → Block 1: Conv2D × 2 + MaxPool + BatchNorm  (32 filters)
    → Block 2: Conv2D × 2 + MaxPool + BatchNorm  (64 filters)
    → Block 3: Conv2D × 2 + MaxPool + BatchNorm  (128 filters)
    → Flatten
    → Dense 512 → Dropout 0.5
    → Dense 256 → Dropout 0.3
    → Dense 2 (Softmax)

| Component | Detail |
|---|---|
| Hidden Activation | ReLU |
| Output Activation | Softmax |
| Regularisation | Dropout (50% + 30%) |
| Normalisation | BatchNormalization after each pooling block |
---
## Smart Callbacks
| Callback | What it does |
|---|---|
| EarlyStopping | Stops if val_loss doesn't improve for 5 epochs. Restores best weights. |
| ReduceLROnPlateau | Halves learning rate if val_loss stalls for 3 epochs. Min LR: 1e-6 |
| ModelCheckpoint | Saves best model weights whenever val_accuracy improves |
---
## Results
### Custom CNN (Baseline)
| Metric | Cat | Dog |
|---|---|---|
| Precision | 0.60 | 0.55 |
| Recall | 0.37 | 0.76 |
| F1-Score | 0.46 | 0.63 |
| **Test Accuracy** | **56.43%** | |
### Transfer Learning — MobileNetV2
Pre-trained on ImageNet. Base layers frozen initially, then top 30 layers fine-tuned with a lower learning rate (1e-5).
| Result | Value |
|---|---|
| **Test Accuracy** | **94.29%** |
> Transfer Learning improved accuracy by ~38% over the custom CNN baseline.
---
## Inference Pipeline
1. **Load Image** — keras.utils.load_img(), resized to 150×150
2. **Preprocess** — convert to array → divide by 255 → add batch dimension
3. **Predict** — model.predict() returns class probabilities
4. **Interpret** — argmax → class label + confidence %
---
## Project Structure

cat-dog-cnn-classifier/
│
├── cat_dog_cnn.ipynb       # Main notebook
├── README.md               # This file

---
## Output 
<img width="1163" height="405" alt="image" src="https://github.com/user-attachments/assets/4ce88d1b-5294-49dd-8faa-94b75b4d0a5c" />
<img width="530" height="581" alt="image" src="https://github.com/user-attachments/assets/a314c194-38fc-485f-96a3-05b8048ea469" />
<img width="961" height="392" alt="image" src="https://github.com/user-attachments/assets/e4ce7e63-00ff-4e00-8883-82af12f3edd3" />
