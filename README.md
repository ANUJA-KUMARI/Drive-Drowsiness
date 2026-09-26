# Driver Drowsiness Detection Using Deep Learning

## Project Overview

This project aims to develop a deep-learning-based system for detecting driver fatigue from facial images.

The current classification task is binary:

- **Active** — driver appears alert/awake
- **Fatigue** — driver appears drowsy/fatigued

The long-term goal is to use the trained model in a **live webcam-based demonstration** that predicts the driver's state in real time.

---

## Team Members

| Name | Roll Number |
|---|---|
| Anuja Kumari | SAU/CS/BTECH(CSE)/2024/16 |
| Shri Mahaveer | SAU/CS/BTECH(CSE)/2024/XX |
| Harsh Raj | SAU/CS/BTECH(CSE)/2024/XX |

> Replace `XX` with the final roll numbers before submission.

---

## Current Project Status

| Component | Status |
|---|---|
| Problem statement | ✅ Completed |
| Dataset acquisition | ✅ Completed |
| Dataset validation/finalization | ✅ Completed |
| Exploratory Data Analysis | ✅ Completed |
| Image preprocessing | ✅ Completed |
| CNN baseline | ⏳ Next |
| Transfer-learning comparison | ⏳ Planned |
| Final evaluation | ⏳ Planned |
| Live webcam demo | ⏳ Planned |

---

## Dataset

The project uses an **UTA-RLDD-derived image dataset** organized into `train`, `val`, and `test` splits with two classes: `active` and `fatigue`.

### Final manifest-defined dataset

The provided manifest files were treated as the authoritative definition of the dataset.

| Split | Active | Fatigue | Total |
|---|---:|---:|---:|
| Train | 3,192 | 3,192 | 6,384 |
| Validation | 912 | 803 | 1,715 |
| Test | 456 | 453 | 909 |
| **Total** | **4,560** | **4,448** | **9,008** |

The dataset is **not included in this GitHub repository** because of its size and distribution considerations. The notebooks expect the dataset to be available locally.

### Expected local dataset structure

```text
data/
└── raw/
    ├── train/
    │   ├── active/
    │   └── fatigue/
    ├── val/
    │   ├── active/
    │   └── fatigue/
    ├── test/
    │   ├── active/
    │   └── fatigue/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

---

## Dataset Validation and Issues Found

During dataset inspection, the downloaded folders contained more image files than were referenced by the supplied manifest files.

The investigation found:

- The folder structure contained additional duplicate files.
- The manifest-defined samples were internally consistent.
- No exact duplicate groups were found across the train, validation, and test manifest-defined splits.
- No conflicting labels were found among the manifest-defined samples.
- No corrupt JPG images were detected in the final manifest-defined set.
- The final working dataset was therefore restricted to the **9,008 samples listed in the manifests**.

This decision prevents accidental inclusion of duplicate folder files and keeps the dataset definition reproducible.

---

## Exploratory Data Analysis

The EDA was performed in:

```text
01_EDA.ipynb
```

The analysis includes:

- Dataset size and split distribution
- Class distribution
- File-type analysis
- Image dimensions
- RGB/mode verification
- Corrupt-image checks
- Sample image inspection
- Random visual inspection
- Brightness analysis
- Class-wise brightness analysis
- Image sharpness analysis
- Aspect-ratio analysis
- Duplicate detection
- Cross-split leakage checks
- Manifest-vs-folder validation

### Major EDA findings

The final images are RGB JPEG files with multiple resolutions and orientations.

Common image dimensions include:

- `1080 × 1920`
- `1080 × 720`
- `1920 × 1080`
- `720 × 1280`
- `640 × 480`

The aspect ratios therefore vary significantly, approximately from `0.56` to `1.78`.

Brightness distributions between Active and Fatigue overlap substantially, indicating that brightness alone should not be sufficient for classification.

Sharpness distributions also overlap, although Fatigue images show somewhat lower typical sharpness.

These observations support the use of a CNN that learns higher-level facial visual features rather than relying on a simple image statistic.

---

## Preprocessing

Image preprocessing is implemented in:

```text
02_Preprocessing.ipynb
```

### Current preprocessing pipeline

```text
Manifest
   ↓
Load image path + label
   ↓
RGB conversion
   ↓
Aspect-ratio-preserving resize
   ↓
Letterbox padding
   ↓
Training augmentation
   ↓
Tensor conversion
   ↓
Normalization
   ↓
PyTorch Dataset
   ↓
PyTorch DataLoader
```

### Input size

All model inputs are converted to:

```text
3 × 224 × 224
```

### Training augmentation

The training pipeline applies mild augmentation such as:

- Horizontal flipping
- Small rotations
- Controlled brightness/contrast variation

Validation and test images do not use random training augmentation.

### Data loading

The current setup uses PyTorch `Dataset` and `DataLoader` objects with a starting batch size of:

```text
32
```

The project is configured to use the available **NVIDIA RTX 3050 4 GB Laptop GPU** through CUDA.

---

## Probable Methodology

The first model will be a **custom CNN baseline**.

```text
Input Image
     ↓
Convolution
     ↓
ReLU
     ↓
Pooling
     ↓
Convolution
     ↓
ReLU
     ↓
Pooling
     ↓
Convolution
     ↓
Feature Representation
     ↓
Fully Connected Layer
     ↓
Active / Fatigue
```

After establishing the baseline, a transfer-learning model such as **ResNet18 or EfficientNet-B0** will be evaluated for comparison.

The final model selection will be based on the measured validation/test performance and computational suitability for the intended live demo.

---

## Evaluation

The final models will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Training/validation loss
- Training/validation accuracy

Special attention will be given to the **Fatigue recall**, because failing to identify fatigued cases is an important error for a driver-monitoring application.

---

## Planned Live Demonstration

The final stage of the project is a real-time webcam demonstration.

```text
Webcam
   ↓
Frame Capture
   ↓
Preprocessing
   ↓
Trained CNN
   ↓
Active / Fatigue
   ↓
Live Status
```

The demonstration is intended to show the trained model working on live camera frames.

---
## Current Repository Structure

```text
Drive-Drowsiness/
├── 01_EDA.ipynb
├── 02_Preprocessing.ipynb
└── README.md

## Planned Repository Structure

The repository will gradually expand to:

```text
Driver-Drowsiness/
│
├── 01_EDA.ipynb
├── 02_Preprocessing.ipynb
├── 03_CNN_Baseline.ipynb
├── 04_Transfer_Learning.ipynb
├── 05_Evaluation.ipynb
│
├── docs/
│   ├── Project_Initiation_Submission_Driver_Drowsiness.docx
│   ├── Driver_Drowsiness_Project_Documentation.docx
│   └── Driver_Drowsiness_EDA_Documentation.docx
│
├── models/
├── app/
├── results/
├── README.md
└── requirements.txt
```

The dataset itself is intentionally excluded from GitHub.

---

## How to Run the Current Notebooks

### 1. Create/activate the project environment

The project was developed in a dedicated Python environment using:

- Python 3.12
- PyTorch
- torchvision
- torchaudio
- Jupyter
- pandas
- NumPy
- Matplotlib
- Seaborn
- Pillow
- OpenCV
- scikit-learn

### 2. Select the Jupyter kernel

Use:

```text
Python (Drowsiness)
```

### 3. Update the dataset path if required

The notebooks currently expect:

```python
DATA_DIR = Path(r"E:\Driver-Drowsiness\data\raw")
```

Change this path if the dataset is stored elsewhere.

### 4. Run the notebooks in order

```text
01_EDA.ipynb
      ↓
02_Preprocessing.ipynb
      ↓
03_CNN_Baseline.ipynb
      ↓
04_Transfer_Learning.ipynb
      ↓
05_Evaluation.ipynb
```

---

## Documentation

Project documentation records the decisions made during development, including dataset validation, EDA findings, preprocessing decisions, model plans, and future work.

The documentation is maintained separately from the notebooks so that the project's reasoning and experimental decisions remain traceable.

---

## Future Work

The remaining development stages are:

1. Implement and train the custom CNN baseline.
2. Track training and validation performance.
3. Evaluate the baseline on the held-out test set.
4. Implement a transfer-learning model.
5. Compare the models using the same evaluation protocol.
6. Perform error analysis.
7. Develop the live webcam demonstration.
8. Document ethical considerations, bias, and limitations.
9. Prepare the final report and presentation.

---

## Course Project Context

This project is being developed as part of a **Neural Networks and Deep Learning** course project.

The project specifically demonstrates the use of:

- Convolutional Neural Networks
- Image preprocessing
- Data augmentation
- GPU-based deep learning
- Model evaluation
- Deep-learning deployment concepts

