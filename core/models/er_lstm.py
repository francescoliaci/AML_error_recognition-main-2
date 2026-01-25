"""
LSTM-based Error Recognition Model

This module implements an LSTM-based baseline for the Supervised Error Recognition task.
The model processes sequences of sub-segment features using bidirectional LSTMs to capture
temporal dependencies in procedural activities

Architecture:
    Input: (batch_size, seq_len, input_dim) - sequence of sub-segment features
    ↓
    Bidirectional LSTM Layer 1 (hidden_size=512)
    ↓
    Dropout (p=0.3)
    ↓
    Bidirectional LSTM Layer 2 (hidden_size=256)
    ↓
    Dropout (p=0.3)
    ↓
    Take final hidden state (512-dim from both directions)
    ↓
    Linear(512, 256) → ReLU → Dropout(0.2)
    ↓
    Linear(256, 1)
    ↓
    Output: Binary prediction (logit, will be passed through BCEWithLogitsLoss)
"""

import torch
import torch.nn as nn
from core.models.blocks import fetch_input_dim


class ErLSTM(nn.Module):
    """
    LSTM-based Error Recognition model
    Args:
        config: configuration object containing model hyperparameters
            - backbone: feature extractor backbone (e.g., 'omnivore', 'slowfast')
            - device: device to run the model on
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        input_dim = fetch_input_dim(config)  # 1024 for Omnivore, 400 for SlowFast

        # bidirectional LSTM layer 1
        # input: (batch_size, seq_len, input_dim)
        # output: (batch_size, seq_len, 512*2)
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0  # no dropout within LSTM (applied manually)
        )

        self.dropout1 = nn.Dropout(0.3)

        # bidirectional LSTM layer 2
        # input: (batch_size, seq_len, 1024)
        # output: (batch_size, seq_len, 256*2)
        self.lstm2 = nn.LSTM(
            input_size=1024,  # 512 * 2 (bidirectional)
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )

        self.dropout2 = nn.Dropout(0.3)

        # classification head
        # Takes the final hidden state (512-dim) and produces a binary prediction
        self.fc1 = nn.Linear(512, 256)  # 256 * 2 (bidirectional)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """
        Forward pass through the LSTM model
        Args:
            x: input tensor of shape (batch_size, seq_len, input_dim)
               where seq_len is the number of sub-segments in a step
        Returns:
            Output tensor of shape (batch_size, 1) containing logits
            for binary classification (will be passed to BCEWithLogitsLoss)
        """
        # Handle NaN values in input
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # First LSTM layer
        # lstm_out1: (batch_size, seq_len, 1024)
        # h_n: (2, batch_size, 512) - final hidden states for both directions
        # c_n: (2, batch_size, 512) - final cell states for both directions
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)

        # Second LSTM layer
        # lstm_out2: (batch_size, seq_len, 512)
        # h_n: (num_directions, batch_size, 256) - final hidden states for both directions
        # c_n: (num_directions, batch_size, 256) - final cell states for both directions
        lstm_out2, (h_n, c_n) = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)

        # Extract final hidden state
        # h_n shape: (num_directions, batch_size, 256) where num_directions=2 for bidirectional
        # Reshape to handle both single and multi-batch cases
        h_n = h_n.view(2, -1, h_n.size(-1))  # (2, batch_size, 256)
        # concatenate forward and backward directions
        final_hidden = torch.cat((h_n[0], h_n[1]), dim=1)  # (batch_size, 512)

        # classification head
        out = self.fc1(final_hidden)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc2(out)  # (batch_size, 1)
        return out