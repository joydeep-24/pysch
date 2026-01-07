"""
Multimodal Emotional Fusion Model Training Script (v3 - Emotion Aligned)
========================================================================

This script trains a mid-level fusion model that integrates:
1. Text emotion (Valence-Arousal)
2. Vision emotion (Valence-Arousal)
3. Conversational context features (4-dim)

Output: Psychological emotional state prediction (5 classes)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'text_dim': 2,      # Valence, Arousal
    'vision_dim': 2,    # Valence, Arousal
    'context_dim': 4,
    'hidden_dims': [64, 32],
    'dropout_rate': 0.3,
    'num_classes': 5,

    'batch_size': 32,
    'num_epochs': 30,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,

    'val_split': 0.2,
    'test_split': 0.1,
    'random_seed': 42,

    'output_dir': 'outputs',
    'model_save_path': 'outputs/fusion_model_v3.pth',
    'metrics_save_path': 'outputs/training_metrics_v3.json',
    'config_save_path': 'outputs/model_config_v3.json'
}

CLASS_LABELS = ["Calm", "Positive", "Neutral", "Stressed", "Distressed"]

# ============================================================================
# EMOTION → VALENCE–AROUSAL MAPPING
# ============================================================================

TEXT_EMOTION_VA = {
    0: (0.8, 0.6), 1: (-0.7, 0.3), 2: (-0.8, 0.8),
    3: (-0.9, 0.9), 4: (0.4, 0.6), 5: (-0.6, 0.4), 6: (0.0, 0.1)
}

VISION_EMOTION_VA = {
    0: (-0.8, 0.8), 1: (-0.6, 0.4), 2: (-0.9, 0.9),
    3: (0.9, 0.6), 4: (-0.7, 0.3), 5: (0.4, 0.6), 6: (0.0, 0.1)
}

def emotion_probs_to_va(emotion_probs, mapping):
    va = np.zeros((emotion_probs.shape[0], 2))
    for i in range(7):
        v, a = mapping[i]
        va[:, 0] += emotion_probs[:, i] * v
        va[:, 1] += emotion_probs[:, i] * a
    return va

# ============================================================================
# WEAK LABEL GENERATION (EMOTION-ALIGNED)
# ============================================================================

def generate_labels(text_va, vision_va, context):
    labels = []
    for (tv, ta), (vv, va), c in zip(text_va, vision_va, context):
        intensity, trend = c[0], c[1]
        valence = 0.6 * tv + 0.4 * vv
        arousal = 0.6 * ta + 0.4 * va

        if valence < -0.5 and arousal > 0.6:
            labels.append(4)  # Distressed
        elif valence > 0.5:
            labels.append(1)  # Positive
        elif intensity < 0.3:
            labels.append(0)  # Calm
        elif valence < -0.2:
            labels.append(3)  # Stressed
        else:
            labels.append(2)  # Neutral

    return np.array(labels)

# ============================================================================
# DATASET
# ============================================================================

class MultimodalEmotionDataset(Dataset):
    def __init__(self, text, vision, context, labels):
        self.text = torch.FloatTensor(text)
        self.vision = torch.FloatTensor(vision)
        self.context = torch.FloatTensor(context)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'text': self.text[idx],
            'vision': self.vision[idx],
            'context': self.context[idx],
            'label': self.labels[idx]
        }

# ============================================================================
# FUSION MODEL
# ============================================================================

class MultimodalFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = CONFIG['text_dim'] + CONFIG['vision_dim'] + CONFIG['context_dim']

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, CONFIG['num_classes'])
        )

    def forward(self, text, vision, context):
        fused = torch.cat([text, vision, context], dim=1)
        return self.net(fused)

# ============================================================================
# TRAINING + VALIDATION
# ============================================================================

def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0, 0, 0
    preds, labels = [], []

    for batch in loader:
        text, vision, context, y = (
            batch['text'].to(device),
            batch['vision'].to(device),
            batch['context'].to(device),
            batch['label'].to(device)
        )

        if train:
            optimizer.zero_grad()

        logits = model(text, vision, context)
        loss = criterion(logits, y)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, p = logits.max(1)
        correct += (p == y).sum().item()
        total += y.size(0)
        preds.extend(p.cpu().numpy())
        labels.extend(y.cpu().numpy())

    acc = 100 * correct / total
    f1 = f1_score(labels, preds, average='macro')
    return total_loss / len(loader), acc, f1

# ============================================================================
# MAIN
# ============================================================================

def main():
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = 1000
    raw_text = np.random.dirichlet(np.ones(7), N)
    raw_vision = np.random.dirichlet(np.ones(7), N)

    text_va = emotion_probs_to_va(raw_text, TEXT_EMOTION_VA)
    vision_va = emotion_probs_to_va(raw_vision, VISION_EMOTION_VA)

    context = np.column_stack([
        np.random.rand(N),
        np.random.uniform(-1, 1, N),
        np.random.rand(N),
        np.random.rand(N)
    ])

    vision_va[np.random.choice(N, int(0.1 * N), replace=False)] = 0
    labels = generate_labels(text_va, vision_va, context)

    idx = np.arange(N)
    tr, te = train_test_split(idx, test_size=CONFIG['test_split'], stratify=labels)
    tr, va = train_test_split(tr, test_size=CONFIG['val_split'], stratify=labels[tr])

    train_ds = MultimodalEmotionDataset(text_va[tr], vision_va[tr], context[tr], labels[tr])
    val_ds = MultimodalEmotionDataset(text_va[va], vision_va[va], context[va], labels[va])
    test_ds = MultimodalEmotionDataset(text_va[te], vision_va[te], context[te], labels[te])

    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=CONFIG['batch_size'])
    test_dl = DataLoader(test_ds, batch_size=CONFIG['batch_size'])

    model = MultimodalFusionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    for epoch in range(CONFIG['num_epochs']):
        tr_l, tr_a, _ = run_epoch(model, train_dl, optimizer, criterion, device, True)
        va_l, va_a, va_f1 = run_epoch(model, val_dl, optimizer, criterion, device, False)

        print(f"Epoch {epoch+1:02d} | Train Acc {tr_a:.2f}% | Val Acc {va_a:.2f}% | Val F1 {va_f1:.4f}")

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), CONFIG['model_save_path'])

    print(f"\nBest Validation F1: {best_f1:.4f}")
    model.load_state_dict(torch.load(CONFIG['model_save_path']))

    _, test_acc, test_f1 = run_epoch(model, test_dl, optimizer, criterion, device, False)
    print(f"Test Accuracy: {test_acc:.2f}% | Test F1: {test_f1:.4f}")

if __name__ == "__main__":
    main()
