# New Baseline Proposal: LSTM-based Error Recognition

## Executive Summary

This document proposes a new **LSTM-based baseline** for the mistake detection task in procedural activities, comparing it with the existing MLP (V1) and Transformer (V2) baselines from the CaptainCook4D paper.

## Background

### Current Baselines

Based on the CaptainCook4D paper and the existing codebase, two baselines are currently implemented:

1. **V1 (MLP)**: Simple Multi-Layer Perceptron
   - Architecture: `Linear(1024, 512) → ReLU → Linear(512, 1)`
   - Input: Pre-extracted features from sub-segments (1-second segments)
   - Limitation: Treats each sub-segment independently, ignoring temporal dependencies

2. **V2 (Transformer)**: Transformer-based encoder
   - Architecture: `TransformerEncoder(d_model=1024, nhead=8, num_layers=1) → MLP(1024, 512, 1)`
   - Input: Sequence of sub-segment features
   - Advantage: Captures relationships between sub-segments using self-attention
   - Current Results (Table 2 from README):
     - Step split: F1=55.39, AUC=75.62
     - Recordings split: F1=40.73, AUC=62.27

### Task Description

**Supervised Error Recognition (SupervisedER)**: Binary classification task to determine if a video segment (corresponding to a procedure step) represents correct or incorrect execution.

**Key Characteristics**:
- Input: Pre-extracted features from Omnivore/SlowFast backbones (1024-dim for Omnivore)
- Each step contains multiple 1-second sub-segments
- Temporal ordering of sub-segments is crucial for understanding the procedure
- Error types include: Technique, Preparation, Temperature, Measurement, and Timing errors

## Proposed Baseline: LSTM-based Error Recognition (V3)

### Motivation

**Why LSTM?**

1. **Sequential Nature**: Unlike Transformers that use self-attention, LSTMs are specifically designed to process sequential data and maintain temporal order through their recurrent connections

2. **Long-term Dependencies**: LSTMs have memory cells that can capture long-range temporal dependencies, which is crucial for detecting errors that manifest over time (e.g., timing errors, temperature errors)

3. **Inductive Bias**: LSTMs have a strong inductive bias towards temporal ordering, which aligns well with procedural activities where step order matters

4. **Computational Efficiency**: LSTMs process sequences sequentially with lower computational cost than Transformers for shorter sequences

5. **Better for Limited Data**: With the CaptainCook4D dataset having limited samples, LSTMs' built-in sequential processing bias may generalize better than Transformers

### Proposed Architecture

```
Input: Sequence of sub-segment features (T × 1024)
  ↓
Bidirectional LSTM Layer 1 (hidden_size=512)
  ↓
Dropout (p=0.3)
  ↓
Bidirectional LSTM Layer 2 (hidden_size=256)
  ↓
Dropout (p=0.3)
  ↓
Take final hidden state (512-dim, from both directions)
  ↓
Linear(512, 256) → ReLU → Dropout(0.2)
  ↓
Linear(256, 1) → Sigmoid
  ↓
Output: Binary prediction (0=correct, 1=error)
```

**Architecture Details**:
- **Bidirectional LSTM**: Processes the sequence both forward and backward, capturing context from both past and future sub-segments
- **Two-layer depth**: Captures hierarchical temporal patterns
- **Dropout regularization**: Prevents overfitting on the limited dataset
- **Final hidden state aggregation**: Summarizes the entire sequence into a fixed representation

### Alternative LSTM Variants to Explore

1. **LSTM with Attention**:
   - Add attention mechanism over LSTM hidden states
   - Allows the model to focus on the most relevant sub-segments

2. **GRU-based Model**:
   - Simpler than LSTM (fewer parameters)
   - May train faster and avoid overfitting

3. **Conv-LSTM**:
   - Add 1D convolution before LSTM
   - Captures local temporal patterns before sequence modeling

## Implementation Plan

### 1. Model Implementation

Create a new file: `core/models/er_lstm.py`

```python
import torch
import torch.nn as nn

class ErLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        input_dim = fetch_input_dim(config)  # 1024 for Omnivore

        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )

        self.dropout1 = nn.Dropout(0.3)

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
        self.fc1 = nn.Linear(512, 256)  # 256 * 2 (bidirectional)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # LSTM layer 1
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)

        # LSTM layer 2
        lstm_out2, (h_n, c_n) = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)

        # Take the final hidden state (concatenated from both directions)
        # h_n shape: (2, batch_size, 256) -> we want (batch_size, 512)
        final_hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)

        # Classification head
        out = self.fc1(final_hidden)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc2(out)

        return out
```

### 2. Integration into Existing Codebase

Modify `base.py` to support LSTM variant:

```python
# In fetch_model function (line 44-56)
elif config.variant == const.LSTM_VARIANT:
    if config.backbone in [const.OMNIVORE, const.RESNET3D, const.X3D, const.SLOWFAST]:
        from core.models.er_lstm import ErLSTM
        model = ErLSTM(config)
```

Add constant in `constants.py`:
```python
LSTM_VARIANT = "LSTM"
```

### 3. Training Configuration

**Hyperparameters**:
- Learning rate: 1e-3 (same as existing baselines)
- Batch size: 1 (due to variable sequence lengths)
- Optimizer: Adam with weight decay 1e-3
- Loss: BCEWithLogitsLoss with pos_weight=2.5 (to handle class imbalance)
- Epochs: 50
- Early stopping: Based on validation AUC

**Training Command**:
```bash
python train_er.py --variant LSTM --backbone omnivore --split step --num_epochs 50
```

### 4. Evaluation Metrics

Following the existing codebase structure, evaluate using:

**Sub-step Level Metrics**:
- Precision, Recall, F1, Accuracy, AUC, PR-AUC

**Step Level Metrics**:
- Precision, Recall, F1, Accuracy, AUC, PR-AUC
- Stratified metrics per error type (already implemented in base.py)

**Threshold Selection**:
- Step split: 0.6
- Recordings split: 0.4

## Expected Advantages of LSTM Baseline

### 1. Temporal Dependency Modeling
- **LSTMs** have explicit memory cells that maintain information across time steps
- Better suited for detecting errors that unfold temporally (e.g., "cooking too long", "forgot to add ingredient")

### 2. Sequential Processing Order
- **LSTMs** process sequences in order (forward/backward), respecting temporal causality
- **Transformers** use self-attention which treats all positions equally (position encoding helps but less explicit)

### 3. Handling Variable-Length Sequences
- **LSTMs** naturally handle variable-length sequences without padding
- **Transformers** require fixed-length inputs or padding, which may introduce noise

### 4. Parameter Efficiency
- **LSTMs** have fewer parameters than Transformers (no multi-head attention)
- May generalize better on the limited CaptainCook4D dataset

### 5. Interpretability
- **LSTM** attention weights (if added) show which sub-segments are important
- Easier to visualize temporal progression of hidden states

## Comparison Framework

### Quantitative Comparison

Create a comparison table with the following structure:

| Model | Split | F1 | Precision | Recall | Accuracy | AUC | PR-AUC | Params | Training Time |
|-------|-------|-----|-----------|--------|----------|-----|--------|--------|---------------|
| MLP (V1) | Step | 24.26 | ? | ? | ? | 75.74 | ? | ~500K | ~1h |
| Transformer (V2) | Step | 55.39 | ? | ? | ? | 75.62 | ? | ~2M | ~3h |
| **LSTM (V3)** | Step | **?** | **?** | **?** | **?** | **?** | **?** | **~1M** | **~2h** |
| MLP (V1) | Recordings | 55.42 | ? | ? | ? | 63.03 | ? | ~500K | ~1h |
| Transformer (V2) | Recordings | 40.73 | ? | ? | ? | 62.27 | ? | ~2M | ~3h |
| **LSTM (V3)** | Recordings | **?** | **?** | **?** | **?** | **?** | **?** | **~1M** | **~2h** |

### Error-Type Specific Analysis

Analyze performance on each error type (leveraging the stratified metrics already in `base.py`):

1. **Technique Error (ID=6)**: Incorrect cooking technique
2. **Preparation Error (ID=2)**: Incorrect ingredient preparation
3. **Temperature Error (ID=3)**: Wrong temperature
4. **Measurement Error (ID=4)**: Wrong quantity
5. **Timing Error (ID=5)**: Wrong duration

**Hypothesis**: LSTMs should perform particularly well on **Timing Errors** and **Temperature Errors** as these require understanding temporal progression.

### Qualitative Comparison

**Strengths of Each Model**:

| Aspect | MLP | Transformer | LSTM |
|--------|-----|-------------|------|
| Temporal Modeling | ❌ None | ✅ Self-attention | ✅✅ Sequential memory |
| Computational Cost | ✅✅ Low | ❌ High | ✅ Medium |
| Training Speed | ✅✅ Fast | ❌ Slow | ✅ Medium |
| Parameter Efficiency | ✅✅ Most efficient | ❌ Many params | ✅ Efficient |
| Long-term Dependencies | ❌ No | ✅ Yes (attention) | ✅ Yes (memory cells) |
| Inductive Bias | ❌ None | 🟡 Position encoding | ✅✅ Sequential processing |
| Handling Variable Lengths | ❌ No | 🟡 Requires padding | ✅ Native support |
| Interpretability | ✅ Simple | 🟡 Attention maps | ✅ Hidden states |

### When to Use Each Model

**Use MLP (V1) when**:
- Computational resources are extremely limited
- Sub-segments are independent
- Need very fast inference

**Use Transformer (V2) when**:
- Sufficient computational resources
- Need to model complex relationships between all sub-segments
- Dataset is large enough to avoid overfitting

**Use LSTM (V3) when**:
- Temporal order is crucial
- Dealing with variable-length sequences
- Need balance between performance and efficiency
- Dataset has strong sequential dependencies

## Ablation Studies

To fully understand the LSTM baseline, conduct the following ablation studies:

### 1. Architecture Variations
- Single-layer vs. two-layer LSTM
- Unidirectional vs. bidirectional
- Different hidden sizes (256, 512, 1024)

### 2. Aggregation Strategies
- Final hidden state (proposed)
- Mean pooling over all hidden states
- Max pooling over all hidden states
- Attention-weighted pooling

### 3. Regularization
- Dropout rates (0.0, 0.2, 0.3, 0.5)
- L2 weight decay (1e-4, 1e-3, 1e-2)

### 4. Alternative RNN Architectures
- GRU (simpler, faster)
- Stacked LSTM (deeper)
- Conv-LSTM (local + global patterns)

## Implementation Checklist

- [ ] Implement `ErLSTM` class in `core/models/er_lstm.py`
- [ ] Add `LSTM_VARIANT` constant to `constants.py`
- [ ] Update `fetch_model()` in `base.py` to support LSTM
- [ ] Test LSTM model on a single batch
- [ ] Train LSTM model on Omnivore features (step split)
- [ ] Evaluate LSTM on test set and compare with MLP/Transformer
- [ ] Analyze per-error-type performance
- [ ] Train LSTM on SlowFast features (optional)
- [ ] Conduct ablation studies
- [ ] Write results section for the report
- [ ] Create visualization of performance comparison
- [ ] Document findings and recommendations

## Expected Outcomes

### Hypothesis 1: Overall Performance
**Prediction**: LSTM will achieve F1 between 45-60% on step split, falling between MLP (24.26%) and Transformer (55.39%)

**Reasoning**:
- Better than MLP due to temporal modeling
- May be slightly worse than Transformer due to sequential bottleneck
- Or may match/exceed Transformer if the dataset benefits from sequential inductive bias

### Hypothesis 2: Error-Type Specific Performance
**Prediction**: LSTM will excel on temporal-sensitive error types

- **Timing Errors**: LSTM >> Transformer > MLP
- **Temperature Errors**: LSTM ≥ Transformer > MLP
- **Technique Errors**: LSTM ≈ Transformer > MLP
- **Measurement Errors**: Transformer ≈ LSTM > MLP
- **Preparation Errors**: Transformer ≈ LSTM > MLP

### Hypothesis 3: Training Efficiency
**Prediction**: LSTM will train faster than Transformer but slower than MLP

- Training time: MLP < LSTM < Transformer
- Convergence: LSTM may converge faster than Transformer due to stronger inductive bias

## Visualization and Analysis

### 1. Learning Curves
Plot training/validation loss and metrics over epochs for all three models

### 2. Confusion Matrices
Show confusion matrices for each model on the test set

### 3. Error Type Analysis
Bar charts comparing F1/Recall/Precision per error type across models

### 4. Sequence Length Analysis
Analyze how performance varies with sequence length (number of sub-segments)

### 5. Hidden State Visualization
Use t-SNE to visualize LSTM hidden states for correct vs. incorrect steps

## Conclusion

This proposal introduces an **LSTM-based baseline** for mistake detection in procedural activities. The LSTM model offers a strong inductive bias for sequential data, explicit temporal modeling through memory cells, and parameter efficiency compared to Transformers.

By implementing and evaluating this baseline, we will gain insights into:
1. The importance of sequential processing for error detection
2. Trade-offs between different temporal modeling approaches
3. Which types of errors benefit most from recurrent architectures
4. Whether the inductive bias of LSTMs helps on limited data

The comparison with existing MLP and Transformer baselines will provide a comprehensive understanding of how different architectures handle the temporal nature of procedural error detection.

## References

1. Peddi, R., et al. (2024). "CaptainCook4D: A dataset for understanding errors in procedural activities." NeurIPS 2024.
2. Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory." Neural Computation.
3. Cho, K., et al. (2014). "Learning phrase representations using RNN encoder-decoder for statistical machine translation." EMNLP 2014.
4. Vaswani, A., et al. (2017). "Attention is all you need." NeurIPS 2017.
5. Flaborea, A., et al. (2024). "PREGO: online mistake detection in PRocedural EGOcentric videos." CVPR 2024.
