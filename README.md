# Wearable Biosignal Transformer Prototype

PyTorch prototype for multimodal wearable time-series modeling with electrodermal activity, blood volume pulse, and temperature signals from the PhysioNet In-Gauge and En-Gage dataset.

The project starts from participant-level wearable CSV files, aligns heterogeneous sensor streams, builds fixed-length windows, pretrains a sequence model with a self-supervised transformation task, and evaluates a TCN plus Transformer classifier with participant-grouped cross-validation.

The current labels are synthetic proxies derived from signal statistics. The reported accuracy should therefore be read as an end-to-end pipeline sanity check, not as validated emotion-recognition performance.

![Pretraining loss](pretraining_loss.png)

## Overview

Wearable biosignals are multimodal, noisy, and unevenly sampled. In this dataset, EDA and temperature are sampled at a lower rate than BVP, so the first part of the workflow is signal alignment: resampling, trimming, filtering, normalization, and fixed-window segmentation.

The implementation keeps the experiment compact and inspectable in `main.py`. It covers the path from raw files to model training and diagnostic plots:

- load EDA, BVP, and TEMP files from the PhysioNet directory structure
- downsample BVP to match the 4 Hz EDA and TEMP rate
- trim all modalities to a shared valid length
- remove invalid rows
- filter EDA, BVP, and TEMP with Butterworth low-pass filters
- normalize each participant sequence with z-score scaling
- cut aligned signals into 60-second non-overlapping windows
- generate synthetic activation and valence-style proxy labels
- pretrain the model on transformation recognition
- fine-tune and evaluate with participant-grouped cross-validation
- save loss, accuracy, and confusion-matrix diagnostics

## Data

The experiment uses the **In-Gauge and En-Gage** wearable dataset from PhysioNet:

```text
physionet.org/files/in-gauge-and-en-gage/1.0.0/class_wearable_data/
```

The current implementation reads three wearable modalities:

| Modality | Source file | Use in the pipeline |
|---|---|---|
| EDA | `EDA.csv` | Electrodermal activity signal |
| BVP | `BVP.csv` | Blood volume pulse signal, downsampled by a factor of 16 |
| TEMP | `TEMP.csv` | Peripheral temperature signal |

Each model input window has shape:

```text
240 samples x 3 modalities
```

This corresponds to 60 seconds at 4 Hz.

## Signal preparation

The preprocessing stage converts participant-level sensor files into aligned windows suitable for sequence modeling.

| Step | Implementation |
|---|---|
| Rate matching | BVP is downsampled to the 4 Hz rate used by EDA and TEMP |
| Alignment | Modalities are trimmed to their shared valid length |
| Cleaning | Invalid values are removed before model input construction |
| Filtering | Low-pass Butterworth filters are applied to each modality |
| Normalization | Each participant sequence is z-scored |
| Windowing | Signals are split into 60-second non-overlapping windows |
| Labels | Activation and valence-style proxy classes are computed from signal statistics |

The proxy labels make the experiment useful for checking preprocessing, batching, model wiring, grouped evaluation, and plotting. They are not a substitute for validated affect annotations.

## Model

The model combines local temporal convolution with a Transformer encoder.

| Component | Implementation |
|---|---|
| Local temporal encoder | Two 1D convolution blocks with batch normalization, ReLU, and dropout |
| Positional information | Sinusoidal positional encoding |
| Sequence encoder | 2-layer Transformer encoder |
| Pooling | Mean pooling across time |
| Classifier | Linear projection to three proxy classes |

Before supervised training, the model is pretrained on a transformation-recognition task. For each input window, the script creates:

1. the original signal
2. a noisy version
3. a magnitude-warped version

The model learns to classify which transformation was applied. This gives a lightweight self-supervised pretraining stage before the proxy-label classifier is fine-tuned.

## Evaluation

The supervised evaluation uses `LeaveOneGroupOut` cross-validation, grouping by participant identifier. This is stricter than a fully random window split, because windows from the same participant do not appear in both train and test folds.

The current script evaluates the first 10,000 windows after preprocessing.

| Output | Current value |
|---|---:|
| Mean accuracy | 0.9356 |
| Standard deviation | 0.0389 |
| Pretraining subset | First 1,000 windows |
| Pretraining epochs | 20 |
| Supervised epochs per fold | 2 |
| Evaluation subset | First 10,000 windows |

![Accuracy distribution](accuracy_boxplot.png)

The confusion matrix shows that most samples belong to the neutral proxy class, so plain accuracy is not enough to characterize model behavior. The result is useful as an execution check for the full pipeline. A stronger evaluation would add class counts, balanced accuracy, macro F1, per-class precision, and per-class recall.

![Confusion matrix](confusion_matrix.png)

## Repository structure

```text
.
├── README.md
├── main.py
├── pretraining_loss.png
├── accuracy_boxplot.png
└── confusion_matrix.png
```

## Run the experiment

Clone the repository:

```bash
git clone https://github.com/gbibbo/transformers_biosignals.git
cd transformers_biosignals
```

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

For Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install numpy pandas scipy scikit-learn torch matplotlib seaborn
```

Download the dataset:

```bash
wget -r -N -c -np https://physionet.org/files/in-gauge-and-en-gage/1.0.0/
```

The expected local path is:

```text
./physionet.org/files/in-gauge-and-en-gage/1.0.0/class_wearable_data/
```

Run:

```bash
python main.py
```

Expected generated files:

```text
pretraining_loss.png
accuracy_boxplot.png
confusion_matrix.png
```

The script also prints the loaded data shape, fold accuracies, mean accuracy, and standard deviation.

## Current status

This is an exploratory research engineering prototype. The main implementation choices are intentionally simple enough to inspect in one script.

The most useful next cleanup would be to turn the prototype into a small experiment package:

```text
src/
├── data.py
├── preprocessing.py
├── models.py
├── train.py
├── evaluate.py
└── plots.py
```

A stronger version would also add:

- `requirements.txt` or `environment.yml`
- command-line arguments for dataset path, window size, batch size, epochs, and output directory
- deterministic seeds
- saved configuration files for each run
- per-class metrics
- explicit class-count reporting
- cleaned fold-level metric bookkeeping
- a small notebook that reproduces the figures from a sample run
- validated downstream labels or a better-defined self-supervised evaluation task

## Citation

If you use or inspect this repository, please cite the dataset and the paper that motivated the exploration.

**Dataset**

Gao, N., Marschall, M., Burry, J., Watkins, S., & Salim, F. (2023). *In-Gauge and En-Gage: Understanding Occupants' Behaviour, Engagement, Emotion, and Comfort Indoors with Heterogeneous Sensors and Wearables* (version 1.0.0). PhysioNet.

**Inspiration paper**

Wu, Y., Daoudi, M., & Amad, A. (2023). *Transformer-based self-supervised multimodal representation learning for wearable emotion recognition*. IEEE Transactions on Affective Computing, 15(1), 157-172.


