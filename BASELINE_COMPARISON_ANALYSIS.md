# Comprehensive Baseline Comparison: MLP vs Transformer vs LSTM

## Executive Summary

This document provides a detailed comparison of three baseline models for the Supervised Error Recognition (SupervisedER) task on the CaptainCook4D dataset:

1. **V1 (MLP)**: Simple Multi-Layer Perceptron baseline
2. **V2 (Transformer)**: Transformer-based encoder baseline
3. **V3 (LSTM)**: Proposed LSTM-based baseline (NEW)

## 1. Task Overview

### Supervised Error Recognition (SupervisedER)

**Goal**: Binary classification of video segments (procedure steps) as correct or incorrect execution.

**Input**:
- Pre-extracted features from video backbones (Omnivore: 1024-dim, SlowFast: 400-dim)
- Each procedure step contains multiple 1-second sub-segments
- Variable sequence lengths (steps have different durations)

**Output**:
- Binary prediction: 0 (correct) or 1 (error)

**Error Types** (from the PDF):
1. **Preparation Error** (ID=2): Incorrect ingredient preparation
2. **Temperature Error** (ID=3): Wrong cooking temperature
3. **Measurement Error** (ID=4): Incorrect quantity/measurements
4. **Timing Error** (ID=5): Wrong duration or timing
5. **Technique Error** (ID=6): Incorrect cooking technique
6. **Order Error**: Wrong sequence of steps
7. **Missing Step**: Omitted required step

**Dataset Splits**:
- **Step split**: Train/val/test split at step level within each recipe
- **Recordings split**: Train/val/test split at recipe level (more challenging)

## 2. Model Architectures

### 2.1 V1: MLP Baseline

**Architecture**:
```
Input: (batch_size, input_dim) - Single sub-segment features
  ↓
Linear(input_dim, 512)
  ↓
ReLU
  ↓
Linear(512, 1)
  ↓
Output: (batch_size, 1) - Binary logit
```

**Key Characteristics**:
- **No temporal modeling**: Each sub-segment processed independently
- **Minimal parameters**: ~500K parameters
- **Training**: Train on sub-segments, test on averaged step predictions
- **Aggregation at inference**: Mean pooling of sub-segment predictions

**Code Location**: [core/models/blocks.py:30-39](core/models/blocks.py#L30-L39)

**Strengths**:
- ✅ Very fast training and inference
- ✅ Minimal computational requirements
- ✅ Simple and interpretable
- ✅ No risk of vanishing/exploding gradients

**Weaknesses**:
- ❌ Ignores temporal dependencies
- ❌ Cannot model sequential patterns
- ❌ Poor performance on timing-related errors
- ❌ Treats beginning and end of step equally

### 2.2 V2: Transformer Baseline

**Architecture**:
```
Input: (batch_size, seq_len, input_dim) - Sequence of sub-segments
  ↓
TransformerEncoder(d_model=1024, nhead=8, num_layers=1, dim_feedforward=2048)
  ↓
Weighted combination of modalities (if multimodal)
  ↓
MLP Decoder: Linear(1024, 512) → ReLU → Linear(512, 1)
  ↓
Output: (batch_size, 1) - Binary logit
```

**Key Characteristics**:
- **Self-attention mechanism**: Models relationships between all sub-segments
- **Position encoding**: Implicit (via learned embeddings or sinusoidal)
- **Parallel processing**: All sub-segments processed simultaneously
- **~2M parameters**

**Code Location**: [core/models/er_former.py](core/models/er_former.py)

**Strengths**:
- ✅ Captures relationships between all sub-segments
- ✅ Attention weights provide interpretability
- ✅ No sequential processing bottleneck
- ✅ State-of-the-art on many sequence tasks

**Weaknesses**:
- ❌ High computational cost (O(n²) complexity)
- ❌ Requires more data to train effectively
- ❌ Weaker inductive bias for temporal order
- ❌ May overfit on small datasets

**Current Performance** (from README.md):

| Split | F1 | AUC |
|-------|-----|-----|
| Step | 55.39 | 75.62 |
| Recordings | 40.73 | 62.27 |

### 2.3 V3: LSTM Baseline (PROPOSED)

**Architecture**:
```
Input: (batch_size, seq_len, input_dim) - Sequence of sub-segments
  ↓
Bidirectional LSTM Layer 1 (hidden_size=512)
  ↓
Dropout (p=0.3)
  ↓
Bidirectional LSTM Layer 2 (hidden_size=256)
  ↓
Dropout (p=0.3)
  ↓
Extract final hidden state: Concat(h_forward, h_backward) → (512-dim)
  ↓
Linear(512, 256) → ReLU → Dropout(0.2)
  ↓
Linear(256, 1)
  ↓
Output: (batch_size, 1) - Binary logit
```

**Key Characteristics**:
- **Sequential processing**: Processes sub-segments in temporal order
- **Memory cells**: Maintains information across time steps
- **Bidirectional**: Captures context from both past and future
- **~1M parameters**

**Code Location**: [core/models/er_lstm.py](core/models/er_lstm.py)

**Strengths**:
- ✅ Explicit temporal modeling through memory cells
- ✅ Strong inductive bias for sequential data
- ✅ Native handling of variable-length sequences
- ✅ Parameter efficient (between MLP and Transformer)
- ✅ Better for limited data (procedural inductive bias)

**Weaknesses**:
- ❌ Sequential bottleneck (cannot parallelize over sequence)
- ❌ May struggle with very long sequences
- ❌ Slower training than MLP
- ❌ Potential vanishing gradient issues (mitigated by LSTM design)

## 3. Detailed Comparison

### 3.1 Architectural Properties

| Property | MLP | Transformer | LSTM |
|----------|-----|-------------|------|
| **Temporal Modeling** | None | Self-attention | Recurrent connections |
| **Sequence Processing** | Independent | Parallel | Sequential |
| **Memory Mechanism** | None | Attention memory | Cell state + hidden state |
| **Inductive Bias** | None | Weak (position encoding) | Strong (sequential) |
| **Position Awareness** | ❌ No | ✅ Yes (encoding) | ✅✅ Yes (implicit) |
| **Long-term Dependencies** | ❌ No | ✅ Yes (direct connections) | ✅ Yes (gating) |
| **Parameters** | ~500K | ~2M | ~1M |
| **Complexity (per forward pass)** | O(d²) | O(n²d) | O(nd²) |

Where:
- `n` = sequence length
- `d` = feature dimension

### 3.2 Training Characteristics

| Aspect | MLP | Transformer | LSTM |
|--------|-----|-------------|------|
| **Training Speed** | ✅✅ Very Fast | ❌ Slow | ✅ Moderate |
| **Convergence** | Fast (simple loss landscape) | May take longer | Moderate |
| **Memory Requirements** | ✅✅ Low | ❌ High | ✅ Moderate |
| **Gradient Flow** | ✅✅ Stable | ✅ Stable (residual) | 🟡 Can be unstable |
| **Hyperparameter Sensitivity** | Low | High | Moderate |
| **Overfitting Risk** | Low (few params) | High (many params) | Moderate |

### 3.3 Inference Characteristics

| Aspect | MLP | Transformer | LSTM |
|--------|-----|-------------|------|
| **Inference Speed** | ✅✅ Very Fast | ❌ Slow | ✅ Fast |
| **Batch Processing** | ✅✅ Easy | ✅ Easy | 🟡 Requires padding |
| **Streaming Capability** | ✅ Yes | ❌ No | ✅✅ Yes (online) |
| **Hardware Requirements** | ✅✅ CPU-friendly | ❌ GPU-heavy | ✅ Moderate |

### 3.4 Task-Specific Suitability

#### For Different Error Types:

| Error Type | MLP | Transformer | LSTM | Reasoning |
|------------|-----|-------------|------|-----------|
| **Technique Error** | 🟡 | ✅ | ✅ | Requires understanding action patterns |
| **Preparation Error** | 🟡 | ✅ | ✅ | May need context of previous steps |
| **Temperature Error** | ❌ | 🟡 | ✅✅ | Temporal: requires tracking heat over time |
| **Measurement Error** | ✅ | ✅ | ✅ | Can be detected in single frame/segment |
| **Timing Error** | ❌ | 🟡 | ✅✅ | Inherently temporal: duration matters |
| **Order Error** | ❌ | ✅ | ✅✅ | Requires understanding sequence |
| **Missing Step** | ❌ | ✅ | ✅ | Requires understanding expected sequence |

**Legend**: ❌ Poor, 🟡 Moderate, ✅ Good, ✅✅ Excellent

#### For Different Sequence Lengths:

| Sequence Length | MLP | Transformer | LSTM |
|-----------------|-----|-------------|------|
| Short (< 5 segments) | ✅ | ✅ | ✅ |
| Medium (5-20 segments) | 🟡 | ✅✅ | ✅ |
| Long (> 20 segments) | ❌ | 🟡 | 🟡 |

**Explanation**:
- **MLP**: Performance degrades as sequence length increases (more averaging)
- **Transformer**: O(n²) attention complexity becomes expensive
- **LSTM**: Sequential processing can lead to information loss in very long sequences

### 3.5 Data Efficiency

| Dataset Size | MLP | Transformer | LSTM |
|--------------|-----|-------------|------|
| Very Small (< 100 samples) | ✅✅ | ❌ | ✅ |
| Small (100-1000 samples) | ✅ | 🟡 | ✅✅ |
| Medium (1000-10000 samples) | ✅ | ✅ | ✅ |
| Large (> 10000 samples) | ✅ | ✅✅ | ✅ |

**CaptainCook4D Dataset Size**: ~1000-2000 procedure steps → **Small to Medium**

**Prediction**: LSTM should excel here due to sequential inductive bias!

## 4. Expected Performance Predictions

### 4.1 Overall Metrics Prediction

**Step Split**:

| Model | Predicted F1 | Predicted AUC | Confidence |
|-------|-------------|---------------|------------|
| MLP | 24.26 (actual) | 75.74 (actual) | Known |
| Transformer | 55.39 (actual) | 75.62 (actual) | Known |
| **LSTM** | **48-58** | **74-78** | High |

**Recordings Split**:

| Model | Predicted F1 | Predicted AUC | Confidence |
|-------|-------------|---------------|------------|
| MLP | 55.42 (actual) | 63.03 (actual) | Known |
| Transformer | 40.73 (actual) | 62.27 (actual) | Known |
| **LSTM** | **45-55** | **62-66** | Medium |

### 4.2 Error-Type Specific Predictions

**Hypothesis**: LSTM will significantly outperform other models on temporal error types.

| Error Type | MLP F1 | Transformer F1 | LSTM F1 (Predicted) |
|------------|---------|----------------|---------------------|
| Technique | ~20-25 | ~50-55 | **52-60** |
| Preparation | ~15-20 | ~45-50 | **48-56** |
| Temperature | ~10-15 | ~40-45 | **50-65** ⬆️ |
| Measurement | ~25-30 | ~55-60 | **54-62** |
| Timing | ~10-15 | ~35-40 | **55-70** ⬆️ |

**Key Prediction**: LSTM will show **20-30% relative improvement** on Temperature and Timing errors compared to Transformer.

### 4.3 Learning Curve Prediction

```
Training Loss Over Epochs:

Epoch 1:  MLP < LSTM < Transformer
Epoch 5:  MLP < LSTM ≈ Transformer
Epoch 10: MLP < LSTM < Transformer
Epoch 20: MLP ≈ LSTM < Transformer
Epoch 50: MLP ≈ LSTM ≈ Transformer
```

**Explanation**:
- **MLP**: Converges very fast (simple model)
- **LSTM**: Moderate convergence (sequential inductive bias helps)
- **Transformer**: Slower initial convergence (more parameters, weaker inductive bias)

## 5. Implementation Guidelines

### 5.1 Training Commands

**Train MLP (V1)**:
```bash
python train_er.py --variant MLP --backbone omnivore --split step --num_epochs 50
```

**Train Transformer (V2)**:
```bash
python train_er.py --variant Transformer --backbone omnivore --split step --num_epochs 50
```

**Train LSTM (V3)**:
```bash
python train_er.py --variant LSTM --backbone omnivore --split step --num_epochs 50
```

**Train LSTM with Attention**:
```bash
python train_er.py --variant LSTM_Attention --backbone omnivore --split step --num_epochs 50
```

**Train GRU (alternative)**:
```bash
python train_er.py --variant GRU --backbone omnivore --split step --num_epochs 50
```

### 5.2 Evaluation Commands

```bash
# Evaluate LSTM on step split (threshold=0.6)
python -m core.evaluate --variant LSTM --backbone omnivore \
    --ckpt checkpoints/error_recognition/LSTM/omnivore/error_recognition_LSTM_omnivore_step_best.pt \
    --split step --threshold 0.6

# Evaluate LSTM on recordings split (threshold=0.4)
python -m core.evaluate --variant LSTM --backbone omnivore \
    --ckpt checkpoints/error_recognition/LSTM/omnivore/error_recognition_LSTM_omnivore_recordings_best.pt \
    --split recordings --threshold 0.4
```

### 5.3 Hyperparameter Recommendations

**Optimal Configuration for LSTM**:

| Hyperparameter | Recommended Value | Alternatives to Try |
|----------------|-------------------|---------------------|
| Hidden Size Layer 1 | 512 | 256, 1024 |
| Hidden Size Layer 2 | 256 | 128, 512 |
| Dropout | 0.3 | 0.2, 0.4, 0.5 |
| Learning Rate | 1e-3 | 5e-4, 1e-4 |
| Weight Decay | 1e-3 | 1e-4, 1e-2 |
| Batch Size | 1 (variable length) | N/A |
| Epochs | 50 | 30-100 |
| Optimizer | Adam | AdamW, SGD |
| Loss Weight (pos_weight) | 2.5 | 2.0, 3.0 |

## 6. Ablation Studies

To fully understand the LSTM baseline, conduct these ablation studies:

### 6.1 Architecture Ablations

| Experiment | Configuration | Expected Impact |
|------------|--------------|-----------------|
| **Baseline** | 2-layer Bi-LSTM (512, 256) | - |
| Unidirectional | Forward-only LSTM | -3 to -5 F1 points |
| Single Layer | 1-layer Bi-LSTM (512) | -2 to -4 F1 points |
| Deeper | 3-layer Bi-LSTM | +1 to -2 F1 points (may overfit) |
| Smaller | 2-layer Bi-LSTM (256, 128) | -2 to -3 F1 points |
| Larger | 2-layer Bi-LSTM (1024, 512) | +0 to +2 F1 points (risk overfitting) |

### 6.2 Aggregation Strategy Ablations

| Strategy | Description | Expected F1 |
|----------|-------------|-------------|
| **Final Hidden State** (proposed) | Use h_n from last LSTM layer | Baseline |
| Mean Pooling | Average all hidden states over sequence | -1 to +1 F1 |
| Max Pooling | Max over all hidden states | -2 to 0 F1 |
| Attention Pooling | Learned attention over hidden states | +1 to +3 F1 |

### 6.3 Regularization Ablations

| Dropout Rate | Expected Training Loss | Expected Val F1 |
|--------------|----------------------|-----------------|
| 0.0 | Lowest | Lower (overfitting) |
| 0.2 | Low | Medium |
| **0.3** (proposed) | Medium | **Highest** |
| 0.5 | Higher | Medium-High |

## 7. Visualization and Analysis Plan

### 7.1 Performance Comparison Plots

1. **Bar Chart: F1 Score by Model**
   - X-axis: Model (MLP, Transformer, LSTM)
   - Y-axis: F1 Score
   - Separate bars for Step and Recordings splits

2. **Grouped Bar Chart: Metrics Comparison**
   - X-axis: Metric (Precision, Recall, F1, AUC)
   - Y-axis: Score
   - Grouped by model

3. **Heatmap: Error Type Performance**
   - Rows: Error types
   - Columns: Models
   - Values: F1 scores
   - Color intensity: Performance level

### 7.2 Training Dynamics

1. **Learning Curves**
   - Plot: Train/Val loss over epochs for all models
   - Plot: F1 score over epochs for all models
   - Shows convergence speed and overfitting

2. **Error Analysis**
   - Confusion matrices for each model
   - Error distribution by error type

### 7.3 Sequence Length Analysis

**Plot**: F1 score vs. sequence length (number of sub-segments)

Expected trend:
- MLP: Performance decreases with sequence length (more averaging dilutes signal)
- Transformer: Relatively stable or slight decrease (attention helps)
- LSTM: Most stable (sequential processing maintains information)

## 8. Discussion and Insights

### 8.1 When to Use Each Model

**Use MLP when**:
- 🚀 Need extremely fast inference (real-time requirements)
- 💰 Computational budget is severely limited
- 🎯 Errors are detectable in single frames/sub-segments
- 📊 Temporal context is not important

**Use Transformer when**:
- 💪 Have sufficient computational resources
- 📈 Dataset is large enough (> 5000 samples)
- 🔍 Need interpretability via attention weights
- 🌐 Errors require global context from entire sequence

**Use LSTM when**:
- ⏰ Temporal order is crucial (timing, temperature errors)
- 📉 Dataset is small to medium (< 5000 samples)
- ⚖️ Need balance between performance and efficiency
- 🎬 Streaming/online inference is required

### 8.2 Key Insights from Comparison

1. **Inductive Bias Matters**: On procedural activity understanding, sequential inductive bias (LSTM) likely helps more than architectural flexibility (Transformer)

2. **Data Efficiency**: With limited data, models with stronger priors (LSTM) generalize better than models requiring more data (Transformer)

3. **Task Alignment**: LSTM's sequential processing naturally aligns with procedural activities where order matters

4. **Error Type Sensitivity**: Different error types benefit from different architectures:
   - Frame-level errors: MLP sufficient
   - Context-dependent errors: Transformer or LSTM
   - Temporal errors: LSTM superior

### 8.3 Future Directions

**Hybrid Approaches**:
1. **Transformer + LSTM**: Use Transformer to encode sub-segments, then LSTM to model sequence
2. **Conv + LSTM**: 1D convolution for local patterns, then LSTM for global sequence
3. **LSTM + Attention**: LSTM with learned attention over hidden states (implemented as variant)

**Advanced Techniques**:
1. **Multi-task Learning**: Jointly predict error type and correctness
2. **Temporal Action Localization**: Identify exact moment when error occurs
3. **Contrastive Learning**: Learn representations that separate correct from incorrect executions

## 9. Conclusion

This comparison provides a comprehensive framework for understanding the trade-offs between MLP, Transformer, and LSTM baselines for mistake detection in procedural activities.

**Key Takeaways**:

1. **No Single Best Model**: Each architecture has its place depending on requirements
2. **LSTM Fills a Gap**: Provides temporal modeling with better data efficiency than Transformers
3. **Sequential Inductive Bias**: Crucial for procedural activity understanding
4. **Error Type Matters**: Some errors are inherently temporal and benefit from LSTM

**Expected Ranking** (Step Split F1):
1. 🥇 Transformer: ~55.4
2. 🥈 LSTM: ~50-58 (predicted)
3. 🥉 MLP: ~24.3

**Expected Ranking** (Recordings Split F1):
1. 🥇 MLP: ~55.4 (surprisingly good!)
2. 🥈 LSTM: ~45-55 (predicted)
3. 🥉 Transformer: ~40.7

The Recordings split results suggest that when errors are sparse across entire recipes, simple averaging (MLP) can work well. But for dense, step-level detection, temporal models (Transformer, LSTM) are essential.

## 10. References

1. Peddi, R., et al. (2024). "CaptainCook4D: A dataset for understanding errors in procedural activities." NeurIPS 2024.
2. Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory." Neural Computation, 9(8), 1735-1780.
3. Vaswani, A., et al. (2017). "Attention is all you need." NeurIPS 2017.
4. Cho, K., et al. (2014). "Learning phrase representations using RNN encoder-decoder." EMNLP 2014.
5. Flaborea, A., et al. (2024). "PREGO: online mistake detection in PRocedural EGOcentric videos." CVPR 2024.
