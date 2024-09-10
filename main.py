import os
import numpy as np
import pandas as pd
from scipy import signal
import seaborn as sns
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import torch
import math
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Route configuration
base_path = "./physionet.org/files/in-gauge-and-en-gage/1.0.0/"
class_data_path = os.path.join(base_path, "class_wearable_data")

def load_and_preprocess_data(class_id, participant_id, window_size=240):  # 60 seconds at 4 Hz
    participant_path = os.path.join(class_data_path, str(class_id), str(participant_id))
    
    try:
        eda = pd.read_csv(os.path.join(participant_path, "EDA.csv"), usecols=['EDA'])
        bvp = pd.read_csv(os.path.join(participant_path, "BVP.csv"), usecols=['BVP'])
        temp = pd.read_csv(os.path.join(participant_path, "TEMP.csv"), usecols=['TEMP'])
    except Exception as e:
        print(f"Error al leer archivos para clase {class_id}, participante {participant_id}: {e}")
        return None
    
    # Convert to float and handle non-numeric values
    eda = pd.to_numeric(eda['EDA'], errors='coerce')
    bvp = pd.to_numeric(bvp['BVP'], errors='coerce')
    temp = pd.to_numeric(temp['TEMP'], errors='coerce')
    
    # Resampling BVP at 4 Hz to match EDA and TEMP
    bvp = bvp.iloc[::16]
    
    # Ensure that all signals are the same length
    min_length = min(len(eda), len(bvp), len(temp))
    eda = eda[:min_length].values
    bvp = bvp[:min_length].values
    temp = temp[:min_length].values
    
    # Delete rows with NaN
    valid_rows = ~np.isnan(eda) & ~np.isnan(bvp) & ~np.isnan(temp)
    data = np.column_stack((eda[valid_rows], bvp[valid_rows], temp[valid_rows]))
    
    if len(data) < window_size:
        print(f"Señal demasiado corta para clase {class_id}, participante {participant_id}")
        return None
    
    # Apply Butterworth filter
    b, a = signal.butter(4, 0.5 / 2, btype='low', fs=4)
    data[:, 0] = signal.filtfilt(b, a, data[:, 0])  # EDA
    data[:, 2] = signal.filtfilt(b, a, data[:, 2])  # TEMP
    
    b, a = signal.butter(4, 2 / 2, btype='low', fs=4)
    data[:, 1] = signal.filtfilt(b, a, data[:, 1])  # BVP
    
    # Normalization z-score
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    # Divide into fixed-size windows
    windows = []
    for i in range(0, len(data) - window_size + 1, window_size):
        windows.append(data[i:i+window_size])
    
    return windows

def generate_synthetic_emotions(data):
    # Simple generation of synthetic labels based on signal patterns
    eda_mean = np.mean(data[:, 0])
    bvp_std = np.std(data[:, 1])
    temp_slope = np.polyfit(range(len(data)), data[:, 2], 1)[0]
    
    if eda_mean > 0.5 and bvp_std > 1.0 and temp_slope > 0:
        return 2  # High activation / Valencia positive
    elif eda_mean < -0.5 and bvp_std < 0.5 and temp_slope < 0:
        return 0  # Low activation / Negative valence
    else:
        return 1  # Neutral State

def apply_transformations(data):
    transformed_data = []
    labels = []
    
    # Original data
    transformed_data.append(data)
    labels.append(0)
    
    # Noise addition
    noise = np.random.normal(0, 0.1, data.shape)
    transformed_data.append(data + noise)
    labels.append(1)
    
    # Magnitude warping
    warping = np.random.normal(1, 0.1, data.shape)
    transformed_data.append(data * warping)
    labels.append(2)
    
    return np.array(transformed_data), np.array(labels)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, num_channels, kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, output_size, kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(output_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        x = x.transpose(1, 2)  # (batch_size, input_size, sequence_length)
        y = self.tcn(x)
        return y.transpose(1, 2)  # (batch_size, sequence_length, output_size)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model

    def forward(self, src):
        # src shape: (batch_size, seq_len, d_model)
        src = src.transpose(0, 1)  # (seq_len, batch_size, d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output.transpose(0, 1)  # (batch_size, seq_len, d_model)

class EmotionRecognitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EmotionRecognitionModel, self).__init__()
        self.tcn = TCN(input_size, hidden_size, 64, 6, 0.1)
        self.transformer = TransformerModel(hidden_size, hidden_size, 4, 2, 128, 0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        x = self.tcn(x)
        # x shape after TCN: (batch_size, sequence_length, hidden_size)
        x = self.transformer(x)
        # x shape after transformer: (batch_size, sequence_length, hidden_size)
        x = torch.mean(x, dim=1)  # Global average pooling
        # x shape after pooling: (batch_size, hidden_size)
        return self.classifier(x)
    
def pretrain(model, dataloader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs} completed")
    
    return model

def train(model, dataloader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# 1. Pre-training loss curve
def pretrain(model, dataloader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(dataloader))
        print(f"Epoch {epoch+1}/{num_epochs} completed")
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), losses)
    plt.title('Pretraining Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('pretraining_loss.png')
    plt.close()
    
    return model

# 2. Box plot of fold accuracies
def plot_accuracy_boxplot(accuracies):
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=accuracies)
    plt.title('Distribution of Accuracies across Folds')
    plt.xlabel('Accuracy')
    plt.savefig('accuracy_boxplot.png')
    plt.close()

# 3. Confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    # Define class labels in English
    class_labels = ['Low Activation\nNegative Valence', 'Neutral\nState', 'High Activation\nPositive Valence']
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create a figure and axis
    plt.figure(figsize=(14, 12))
    
    # Plot heatmap
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=False,
                xticklabels=class_labels, yticklabels=class_labels)
    
    # Improve aesthetics
    plt.title('Confusion Matrix', fontsize=18)
    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    
    # Add text annotations with more space between count and percentage
    total = np.sum(cm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.35, f'{cm[i, j]}', 
                     ha='center', va='center', fontsize=14, fontweight='bold')
            plt.text(j + 0.5, i + 0.65, f'({cm[i, j]/total:.1%})',
                     ha='center', va='center', fontsize=12)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()

# Loading and preprocessing data
all_data = []
all_labels = []
participant_ids = []

for class_id in os.listdir(class_data_path):
    if class_id.isdigit():
        class_path = os.path.join(class_data_path, class_id)
        if os.path.isdir(class_path):
            for participant_id in os.listdir(class_path):
                if participant_id.isdigit() or participant_id == 'teacher':
                    windows = load_and_preprocess_data(class_id, participant_id)
                    if windows is not None and len(windows) > 0:
                        for window in windows:
                            label = generate_synthetic_emotions(window)
                            all_data.append(window)
                            all_labels.append(label)
                            participant_ids.append(participant_id)

if not all_data:
    raise ValueError("No se encontraron datos válidos en ninguna clase/participante")

X = np.array(all_data)
y = np.array(all_labels)

print(f"Forma de los datos cargados: {X.shape}")

# Create model
model = EmotionRecognitionModel(input_size=3, hidden_size=64, num_classes=3).to(device)

# Self-supervised pre-training
pretrain_data = []
pretrain_labels = []
for data in X[:1000]:  # Limit to the first 1000 samples for pre-training
    transformed_data, transform_labels = apply_transformations(data)
    pretrain_data.extend(transformed_data)
    pretrain_labels.extend(transform_labels)

pretrain_data = np.array(pretrain_data)
pretrain_labels = np.array(pretrain_labels)

print(f"Pretrain data shape: {pretrain_data.shape}")
print(f"Unique pretrain labels: {np.unique(pretrain_labels)}")

pretrain_dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(pretrain_data).to(device), 
    torch.LongTensor(pretrain_labels).to(device)
)
pretrain_dataloader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=32, shuffle=True)

model = pretrain(model, pretrain_dataloader, num_epochs=20, device=device)

# Supervised training with cross-validation LOSO
loso = LeaveOneGroupOut()
accuracies = []

# Use only a fraction of the data for supervised training.
X_subset = X[:10000]
y_subset = y[:10000]
participant_ids_subset = participant_ids[:10000]
accuracies = []
all_y_true = []
all_y_pred = []

for train_index, test_index in loso.split(X_subset, y_subset, groups=participant_ids_subset):
    X_train, X_test = X_subset[train_index], X_subset[test_index]
    y_train, y_test = y_subset[train_index], y_subset[test_index]
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train).to(device), 
        torch.LongTensor(y_train).to(device)
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test).to(device), 
        torch.LongTensor(y_test).to(device)
    )
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model_copy = copy.deepcopy(model)
    model_copy = train(model_copy, train_dataloader, num_epochs=2, device=device)
    accuracy = evaluate(model_copy, test_dataloader, device=device)
    accuracies.append(accuracy)

    accuracy = evaluate(model_copy, test_dataloader, device=device)
    accuracies.append(accuracy)
    y_true, y_pred = [], []
    model_copy.eval()
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_copy(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    all_y_true.extend(y_true)
    all_y_pred.extend(y_pred)
    print(f"Fold completed. Accuracy: {accuracy}")

plot_accuracy_boxplot(accuracies)
plot_confusion_matrix(all_y_true, all_y_pred)

print(f"Mean accuracy: {np.mean(accuracies)}")
print(f"Std accuracy: {np.std(accuracies)}")
