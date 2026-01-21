"""
LSTM-based Error Recognition Model

This module implements an LSTM-based baseline for the Supervised Error Recognition task.
The model processes sequences of sub-segment features using bidirectional LSTMs to capture
temporal dependencies in procedural activities.

Architecture:
    Input: (batch_size, seq_len, input_dim) - Sequence of sub-segment features
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
    LSTM-based Error Recognition model.

    This model uses a two-layer bidirectional LSTM to process sequences of
    pre-extracted video features (sub-segments) and predicts whether a step
    contains an error.

    Args:
        config: Configuration object containing model hyperparameters
            - backbone: Feature extractor backbone (e.g., 'omnivore', 'slowfast')
            - device: Device to run the model on
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        input_dim = fetch_input_dim(config)  # 1024 for Omnivore, 400 for SlowFast

        # Bidirectional LSTM layer 1
        # Input: (batch_size, seq_len, input_dim)
        # Output: (batch_size, seq_len, 512*2)
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0  # No dropout within LSTM (we apply it manually)
        )

        self.dropout1 = nn.Dropout(0.3)

        # Bidirectional LSTM layer 2
        # Input: (batch_size, seq_len, 1024)
        # Output: (batch_size, seq_len, 256*2)
        self.lstm2 = nn.LSTM(
            input_size=1024,  # 512 * 2 (bidirectional)
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )

        self.dropout2 = nn.Dropout(0.3)

        # Classification head
        # Takes the final hidden state (512-dim) and produces a binary prediction
        self.fc1 = nn.Linear(512, 256)  # 256 * 2 (bidirectional)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """
        Forward pass through the LSTM model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
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
        # h_n: (2, batch_size, 256) - final hidden states for both directions
        # c_n: (2, batch_size, 256) - final cell states for both directions
        lstm_out2, (h_n, c_n) = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)

        # Extract final hidden state
        # h_n shape: (2, batch_size, 256) where:
        #   h_n[0] = final hidden state of forward LSTM
        #   h_n[1] = final hidden state of backward LSTM
        # We concatenate them to get (batch_size, 512)
        final_hidden = torch.cat((h_n[0, :, :], h_n[1, :, :]), dim=1)

        # Classification head
        out = self.fc1(final_hidden)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc2(out)  # (batch_size, 1)

        return out


class ErLSTMWithAttention(nn.Module):
    """
    LSTM-based Error Recognition model with attention mechanism.

    This variant adds an attention layer over the LSTM hidden states,
    allowing the model to focus on the most relevant sub-segments.

    Args:
        config: Configuration object containing model hyperparameters
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        input_dim = fetch_input_dim(config)

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Attention mechanism
        # Query: final hidden state, Keys/Values: all hidden states
        self.attention = nn.MultiheadAttention(
            embed_dim=1024,  # 512 * 2 (bidirectional)
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )

        # Classification head
        self.fc1 = nn.Linear(1024, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        """
        Forward pass with attention over LSTM hidden states.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch_size, 1) containing logits
        """
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch_size, seq_len, 1024)

        # Self-attention over LSTM outputs
        # Use the sequence as query, key, and value
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Take mean over sequence dimension
        # Alternative: take final state or max pooling
        context = torch.mean(attn_out, dim=1)  # (batch_size, 1024)

        # Classification head
        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class ErGRU(nn.Module):
    """
    GRU-based Error Recognition model.

    GRU is a simpler alternative to LSTM with fewer parameters.
    It may train faster and generalize better on limited data.

    Args:
        config: Configuration object containing model hyperparameters
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        input_dim = fetch_input_dim(config)

        # Bidirectional GRU layers
        self.gru1 = nn.GRU(
            input_size=input_dim,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )

        self.dropout1 = nn.Dropout(0.3)

        self.gru2 = nn.GRU(
            input_size=1024,  # 512 * 2
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )

        self.dropout2 = nn.Dropout(0.3)

        # Classification head
        self.fc1 = nn.Linear(512, 256)  # 256 * 2
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """
        Forward pass through the GRU model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch_size, 1) containing logits
        """
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # First GRU layer
        gru_out1, _ = self.gru1(x)
        gru_out1 = self.dropout1(gru_out1)

        # Second GRU layer
        gru_out2, h_n = self.gru2(gru_out1)
        gru_out2 = self.dropout2(gru_out2)

        # Extract final hidden state
        final_hidden = torch.cat((h_n[0, :, :], h_n[1, :, :]), dim=1)

        # Classification head
        out = self.fc1(final_hidden)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc2(out)

        return out
