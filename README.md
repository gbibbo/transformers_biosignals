# Wearable Biosignal Transformer Prototype

PyTorch prototype for multimodal wearable time-series modeling using electrodermal activity, blood volume pulse, and temperature signals from the PhysioNet In-Gauge and En-Gage dataset.

The repository starts from raw participant-level wearable CSV files, aligns heterogeneous sensor streams, builds fixed-length windows, pretrains a sequence model with a self-supervised transformation task, and evaluates a TCN plus Transformer classifier with participant-grouped cross-validation.

This is an exploratory research engineering project. It is not a validated emotion-recognition system. The current labels are synthetic proxies derived from signal statistics, so the reported accuracy should be read as a pipeline sanity check, not as affect-recognition performance.

![Confusion Matrix](confusion_matrix.png)

## Overview

Wearable physiological data is noisy, multimodal, and unevenly sampled. EDA and temperature are available at lower sampling rates than BVP, and any useful modeling workflow needs to make those streams comparable before training a model.

This project implements that workflow in a compact script:

- loads EDA, BVP, and TEMP files from the PhysioNet directory structure
- downsamples BVP to match the 4 Hz EDA and TEMP rate
- trims all modalities to a shared valid length
- removes invalid rows
- filters EDA, BVP, and TEMP with Butterworth low-pass filters
- normalizes each participant sequence with z-score scaling
- cuts the aligned signals into 60-second non-overlapping windows
- generates synthetic activation and valence-style proxy labels
- pretrains the model on transformation classification
- fine-tunes and evaluates with grouped cross-validation by participant identifier
- saves diagnostic plots for inspection

## Data

The experiment uses the **In-Gauge and En-Gage** wearable dataset from PhysioNet:

```text
physionet.org/files/in-gauge-and-en-gage/1.0.0/class_wearable_data/
```

The current implementation reads three wearable modalities:

| Modality | Source file | Role in the pipeline |
|---|---|---|
| EDA | `EDA.csv` | Electrodermal activity signal |
| BVP | `BVP.csv` | Blood volume pulse signal, downsampled by a factor of 16 |
| TEMP | `TEMP.csv` | Peripheral temperature signal |

Each window has shape:

```text
240 samples x 3 modalities
```

This corresponds to 60 seconds at 4 Hz.

## Modeling workflow

The model combines a temporal convolutional front end with a Transformer encoder:

| Component | Implementation |
|---|---|
| Local temporal encoder | Two 1D convolution blocks with batch normalization, ReLU, and dropout |
| Sequence encoder | Sinusoidal positional encoding plus 2-layer Transformer encoder |
| Pooling | Mean pooling across time |
| Classifier | Linear projection to three proxy classes |

Before supervised training, the model is pretrained on a simple transformation-recognition task. For each input window, the script creates:

1. the original signal
2. a noisy version
3. a magnitude-warped version

The model learns to classify which transformation was applied.

![Pretraining Loss](pretraining_loss.png)

## Evaluation

The supervised evaluation uses `LeaveOneGroupOut` cross-validation, grouping by participant identifier. This avoids a fully random split where windows from the same participant could appear in both train and test sets.

The current script evaluates the first 10,000 windows after preprocessing.

| Output | Current value |
|---|---:|
| Mean accuracy | 0.9356 |
| Standard deviation | 0.0389 |
| Pretraining subset | First 1,000 windows |
| Pretraining epochs | 20 |
| Supervised epochs per fold | 2 |
| Evaluation subset | First 10,000 windows |

![Accuracy Distribution](accuracy_boxplot.png)

The confusion matrix shows that most samples belong to the neutral proxy class. For that reason, the accuracy is useful for checking that the pipeline runs end to end, but it should not be interpreted as evidence of reliable emotion recognition.

## Repository structure

```text
.
├── README.md
├── main.py
├── confusion_matrix.png
├── accuracy_boxplot.png
└── pretraining_loss.png
```

## Run the experiment

Clone the repository:

```bash
git clone https://github.com/<your-username>/transformers_biosignals.git
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

## Current scope

This repository is intentionally framed as a prototype. The main constraints are:

- the labels are synthetic proxies, not validated affect annotations
- the setup is not a faithful reproduction of the reference paper
- only EDA, BVP, and TEMP are used
- the script is monolithic and would benefit from modularization
- the evaluation would be stronger with balanced accuracy, macro F1, per-class precision, per-class recall, and explicit class counts
- the experiment should add fixed random seeds and saved run metadata for stricter reproducibility
- the current evaluation bookkeeping should be cleaned so fold metrics are recorded once per fold

## Next improvements

The most useful next step would be to turn the prototype into a cleaner experiment package:

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
- a lightweight notebook that reproduces the figures from a small sample
- real downstream labels or a better-defined self-supervised evaluation task

## Citation

If you use or inspect this repository, please cite the dataset and the paper that motivated the exploration.

**Dataset**

Gao, N., Marschall, M., Burry, J., Watkins, S., & Salim, F. (2023). *In-Gauge and En-Gage: Understanding Occupants' Behaviour, Engagement, Emotion, and Comfort Indoors with Heterogeneous Sensors and Wearables* (version 1.0.0). PhysioNet.

**Inspiration paper**

Wu, Y., Daoudi, M., & Amad, A. (2023). *Transformer-based self-supervised multimodal representation learning for wearable emotion recognition*. IEEE Transactions on Affective Computing, 15(1), 157-172.

