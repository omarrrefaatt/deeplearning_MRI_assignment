# Clinical MRI Classification — Medical Report

**Student Name:** Omar Ahmed Mohamed Refaat
**Student ID:** 1210106  
**Course:** Deep Learning in Medicine (SBES361)

---

## Dataset

**Brain Tumor MRI Dataset** — [Kaggle (masoudnickparvar)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

Images are split into `Training/` and `Testing/` directories, each containing four class sub-folders. The training set is further divided 80/20 into train and validation splits at load time.

| Split            | Source                       |
| ---------------- | ---------------------------- |
| Train (80%)      | `Training/`                  |
| Validation (20%) | `Training/` (held-out split) |
| Test             | `Testing/`                   |

**Input dimensions:** 224 × 224 × 3 (RGB) — **Batch size:** 32

### Preprocessing & Augmentation

All images are rescaled to `[0, 1]`. The training pipeline additionally applies on-the-fly augmentation:

| Transform              | Detail                       |
| ---------------------- | ---------------------------- |
| Random Horizontal Flip | Applied during training only |
| Random Rotation        | ±10° (factor = 0.1)          |
| Random Zoom            | ±10% (factor = 0.1)          |

Validation and test sets receive normalisation only (no augmentation). All pipelines use `tf.data` caching and prefetching (`AUTOTUNE`) for performance.

---

## Model Architecture

The custom CNN (`brain_tumor_classifier`) accepts **224×224 RGB** MRI images and is organised as four progressively deeper convolutional blocks followed by a regularised fully-connected classifier head.

```
Input (224 × 224 × 3)
  ──────────────────────────────────────────────────────────────────
  Block 1 │ Conv2D(32,  3×3, ReLU, same) → BatchNorm → MaxPool(2×2)  →  112×112×32
  Block 2 │ Conv2D(64,  3×3, ReLU, same) → BatchNorm → MaxPool(2×2)  →   56×56×64
  Block 3 │ Conv2D(128, 3×3, ReLU, same) → BatchNorm → MaxPool(2×2)  →   28×28×128
  Block 4 │ Conv2D(256, 3×3, ReLU, same) → BatchNorm → MaxPool(2×2)  →   14×14×256
  ──────────────────────────────────────────────────────────────────
           │ GlobalAveragePooling2D                                    →   256
           │ Dense(256, ReLU, L2 λ=0.001)
           │ Dropout(0.5)
           │ Dense(4, Softmax)                                         →   4 classes
```

**Total trainable parameters: ~456,196 (~1.74 MB)**

### Regularisation Techniques Applied

| Technique                | Detail                                                          |
| ------------------------ | --------------------------------------------------------------- |
| Dropout                  | Rate = 0.5 on the dense layer                                   |
| L2 Weight Regularisation | λ = 0.001 on the dense layer                                    |
| Batch Normalisation      | After every convolutional block                                 |
| Data Augmentation        | Random horizontal flip, rotation (±10°), zoom (±10%)            |
| EarlyStopping            | Patience = 5 epochs, monitors `val_loss`, restores best weights |
| ReduceLROnPlateau        | Factor = 0.5, patience = 3 epochs, min LR = 1×10⁻⁶              |

**Parameter Count Justification:** The ~456 K trainable parameters is appropriate for this dataset because the combined use of Dropout (0.5), L2 regularisation, BatchNormalisation, data augmentation, and EarlyStopping substantially reduces effective model capacity and prevents the network from memorising training examples.

---

## Training Configuration

| Setting                | Value                               |
| ---------------------- | ----------------------------------- |
| Optimiser              | Adam (lr = 0.001)                   |
| Loss                   | `SparseCategoricalCrossentropy`     |
| Metric                 | Accuracy                            |
| Max epochs             | 30                                  |
| EarlyStopping patience | 5 (monitors `val_loss`)             |
| LR reduction factor    | 0.5 (patience = 3, min LR = 1×10⁻⁶) |

Training curves (loss and accuracy for both train and validation) are saved to `training_curves.png`.

---

## Per-Class Recall

| Class      | Recall |
| ---------- | ------ |
| glioma     | 0.59   |
| meningioma | 0.52   |
| notumor    | 1.00   |
| pituitary  | 0.99   |

> **Clinical Note:** The `notumor` class achieved a recall of **≈1.00**, meaning the model correctly identifies virtually all tumour-free patients. A False Negative on this class (missing a tumour) is the most critical clinical failure mode, and the model performs strongly here. However, recall for `glioma` (0.59) and `meningioma` (0.52) is considerably lower, indicating meaningful under-detection of these tumour types that warrants further investigation.

---

## Error Analysis

**Most confused pair: `meningioma` ↔ `glioma`**

The confusion matrix reveals that the largest source of misclassification is between the meningioma and glioma classes. This overlap has a clear biological and radiological basis.

Both tumour types occur within the cranial cavity and can exhibit similar appearances on standard MRI sequences depending on stage and grade. They may present with comparable signal intensities and can both produce mass effect and surrounding oedema.

While meningiomas are typically _extra-axial_ tumours with dural attachment — often showing a characteristic "dural tail" sign and well-defined, rounded margins — these features are not always visible, especially in atypical cases. Gliomas, by contrast, are _intra-axial_ and infiltrative, with poorly defined, irregular borders. However, low-grade gliomas can appear relatively well-circumscribed, erasing the textbook distinction.

Because the model operates solely on single-sequence RGB-represented MRI slices without explicit anatomical priors, it has limited ability to consistently exploit the extra-axial vs. intra-axial distinction. Providing multi-sequence input (e.g., FLAIR for oedema mapping, DWI for cellularity, post-contrast T1 for enhancement pattern) would supply the additional radiological context needed to resolve this class pair more reliably.

---
