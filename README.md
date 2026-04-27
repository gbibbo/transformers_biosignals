# Multimodal Transformer Pipeline for Wearable Biosignals

Research engineering prototype for wearable physiological time-series modeling with PyTorch. The project loads EDA, BVP, and temperature signals from the PhysioNet In-Gauge and En-Gage dataset, preprocesses the modalities into aligned windows, trains a TCN plus Transformer model, and reports evaluation artifacts with explicit caveats about the use of synthetic proxy labels.

**Evidence of work:** multimodal signal preprocessing, time-series modeling, self-supervised pretraining, grouped evaluation, and transparent reporting of limitations.

## Problem

Wearable biosignals are noisy, heterogeneous, and sampled at different rates. A useful ML pipeline must do more than define a model. It needs to load raw sensor files, align modalities, handle missing or invalid samples, define trainable targets, evaluate outputs, and make limitations visible.

This repository explores that workflow on physiological signals from the In-Gauge and En-Gage dataset. It is not a validated emotion recognition system. Its value is the end-to-end research engineering implementation: from raw wearable signals to model training, cross-validation, and diagnostic plots.

## What I built

| Layer | Implementation |
|---|---|
| Data ingestion | Loads participant-level wearable CSV files for EDA, BVP, and TEMP from the PhysioNet directory structure. |
| Signal alignment | Downsamples BVP to match the 4 Hz EDA and TEMP rate, trims modalities to a shared length, and removes rows with invalid values. |
| Preprocessing | Applies Butterworth low-pass filtering, z-score normalization, and fixed 60-second non-overlapping windows. |
| Proxy label generation | Creates three synthetic activation and valence-style labels using EDA mean, BVP variability, and temperature slope. |
| Self-supervised pretraining | Trains the model to classify transformations: original signal, additive noise, and magnitude warping. |
| Model architecture | Combines a TCN front-end, sinusoidal positional encoding, Transformer encoder layers, global average pooling, and a linear classifier. |
| Evaluation workflow | Uses grouped cross-validation by participant identifier on a 10,000-window subset and saves plots for inspection. |
| Reporting artifacts | Produces a confusion matrix, fold accuracy boxplot, and pretraining loss curve. |

## Tech stack

| Area | Tools |
|---|---|
| Deep learning | PyTorch |
| Data handling | NumPy, pandas |
| Signal processing | SciPy |
| Evaluation | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Dataset | PhysioNet In-Gauge and En-Gage wearable data |

## Result / evidence

The current run reports:

| Metric | Value |
|---|---:|
| Mean grouped CV accuracy | 0.9356 |
| Standard deviation | 0.0389 |
| Model input modalities | EDA, BVP, TEMP |
| Window size | 240 samples, corresponding to 60 seconds at 4 Hz |
| Pretraining epochs | 20 |
| Supervised training epochs per fold | 2 |
| Evaluation subset | First 10,000 windows |

These numbers should be read as a pipeline sanity check, not as real emotion-recognition performance. The labels are synthetic and derived from the input signals through simple rules. The confusion matrix also shows that the proxy-label distribution is dominated by the neutral class, so the high accuracy is not sufficient evidence of robust affect recognition.

The important engineering evidence is that the repository implements the full workflow and produces auditable outputs.

### Confusion matrix

The confusion matrix makes the class imbalance visible. The model mostly predicts the neutral proxy class, which is why the metric should not be overinterpreted.

![Confusion Matrix](confusion_matrix.png)

### Accuracy distribution across grouped folds

The fold-level accuracy distribution shows the spread of the grouped evaluation runs.

![Accuracy Distribution](accuracy_boxplot.png)

### Self-supervised pretraining loss

The pretraining objective converges on the transformation-classification task. This confirms that the model learns the proxy pretraining task, but it does not validate real emotion recognition.

![Pretraining Loss](pretraining_loss.png)

## How to run / demo

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/transformers_biosignals.git
cd transformers_biosignals
```

### 2. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

For Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install numpy pandas scipy scikit-learn torch matplotlib seaborn
```

### 4. Download the dataset

The script expects the PhysioNet dataset at:

```text
./physionet.org/files/in-gauge-and-en-gage/1.0.0/
```

Download it with:

```bash
wget -r -N -c -np https://physionet.org/files/in-gauge-and-en-gage/1.0.0/
```

After download, the repository should contain:

```text
physionet.org/files/in-gauge-and-en-gage/1.0.0/class_wearable_data/
```

### 5. Run the experiment

```bash
python main.py
```

Expected outputs:

```text
pretraining_loss.png
accuracy_boxplot.png
confusion_matrix.png
```

The script also prints the loaded data shape, fold accuracies, mean accuracy, and standard deviation.

## Repository structure

```text
.
├── README.md
├── main.py
├── confusion_matrix.png
├── accuracy_boxplot.png
└── pretraining_loss.png
```

## What this demonstrates for ML roles

This project is most relevant as evidence for roles involving applied ML, physiological or sensor data, audio and time-series modeling, and research prototyping.

It demonstrates that I can:

- Translate a paper-inspired idea into an executable PyTorch prototype.
- Build a preprocessing pipeline for heterogeneous sensor streams.
- Combine temporal convolution and Transformer encoder blocks for sequence modeling.
- Design a self-supervised pretraining task for time-series data.
- Use grouped evaluation to reduce overly optimistic random-split reporting.
- Produce visual diagnostics and communicate metric limitations honestly.

## Known limitations

This repository is intentionally framed as a prototype. The following limitations are important:

- It does not use validated emotional annotations.
- The labels are synthetic proxies derived from the same signal statistics used for learning.
- The reported accuracy is not comparable to supervised emotion-recognition benchmarks.
- Only three modalities are used: EDA, BVP, and TEMP.
- The implementation is a simplified exploration, not a faithful reproduction of the referenced paper.
- The current script is monolithic and would benefit from configuration files, experiment tracking, fixed random seeds, and a `requirements.txt`.
- The grouped split uses the participant identifier as implemented in the dataset traversal. A production-grade version should audit subject identity handling across classes and sessions.

## Next steps

Strong next engineering improvements would be:

1. Replace synthetic proxy labels with validated annotations or a clearly defined downstream task.
2. Add `requirements.txt` or `environment.yml`.
3. Split `main.py` into data, models, training, evaluation, and plotting modules.
4. Add CLI arguments for dataset path, window size, epochs, batch size, and output directory.
5. Report balanced accuracy, macro F1, per-class precision and recall, and class counts.
6. Add deterministic seeds and save experiment metadata.
7. Add a lightweight demo notebook that loads a small sample and reproduces the figures.

## Citation

If you use or inspect this repository, please cite the dataset and the paper that motivated the exploration.

**Dataset**

Gao, N., Marschall, M., Burry, J., Watkins, S., & Salim, F. (2023). *In-Gauge and En-Gage: Understanding Occupants' Behaviour, Engagement, Emotion, and Comfort Indoors with Heterogeneous Sensors and Wearables* (version 1.0.0). PhysioNet.

**Inspiration paper**

Wu, Y., Daoudi, M., & Amad, A. (2023). *Transformer-based self-supervised multimodal representation learning for wearable emotion recognition*. IEEE Transactions on Affective Computing, 15(1), 157-172.
