# LSTM Baseline for Mistake Detection in Procedural Activities

This project implements a new baselines for supervised error recognition in cooking videos: an LSTM that processes sequences bidirectionally

## Quick Start

The easiest way to get started is using the Colab notebook:

📓 **[colab_lstm_training.ipynb](colab_lstm_training.ipynb)**

The notebook handles environment setup, feature extraction, training, and evaluation across all three models

## Architecture

The LSTM processes variable-length sequences of video features through two bidirectional LSTM layers, followed by a classification head. The model uses dropout for regularization and outputs a binary prediction (correct vs. error).

```
Input (T × 1024) → BiLSTM(512) → Dropout → BiLSTM(256) → Dropout
  → Linear(512, 256) → ReLU → Dropout → Linear(256, 1) → Output
```

## Project Structure

```
├── core/
│   ├── models/
│   │   ├── er_lstm.py          # LSTM implementation
│   │   ├── er_former.py        # Transformer baseline
│   │   └── blocks.py           # MLP baseline
│   ├── evaluate.py
│   └── config.py
├── dataloader/
│   ├── CaptainCookStepDataset.py
│   └── CaptainCookSubStepDataset.py
├── train_er.py
├── colab_lstm_training.ipynb
└── README.md
```

## Configuration

```python
learning_rate = 1e-3
batch_size = 1
optimizer = Adam(weight_decay=1e-3)
loss = BCEWithLogitsLoss(pos_weight=2.5)
epochs = 50
```

---

**Last updated**: January 2026
