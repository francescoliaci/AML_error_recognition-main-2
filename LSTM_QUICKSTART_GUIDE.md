# LSTM Baseline Quick Start Guide

This guide provides step-by-step instructions for implementing, training, and evaluating the LSTM baseline for mistake detection.

## Prerequisites

Ensure you have:
- ✅ Python environment with PyTorch installed
- ✅ Downloaded pre-extracted features (Omnivore/SlowFast)
- ✅ Placed features in `data/` directory
- ✅ Downloaded annotations (already in `annotations/` directory)

## Implementation Status

### Files Created/Modified

1. ✅ **[core/models/er_lstm.py](core/models/er_lstm.py)** - LSTM model implementations
   - `ErLSTM`: Main LSTM baseline
   - `ErLSTMWithAttention`: LSTM + attention variant
   - `ErGRU`: GRU alternative

2. ✅ **[constants.py](constants.py)** - Added new constants
   - `LSTM_VARIANT = "LSTM"`
   - `LSTM_ATTENTION_VARIANT = "LSTM_Attention"`
   - `GRU_VARIANT = "GRU"`

3. ✅ **[base.py](base.py)** - Updated model factory
   - Added LSTM variant support in `fetch_model()`

## Quick Start: Training LSTM

### Step 1: Verify Installation

```bash
# Navigate to project directory
cd /Users/francescoliaci/Desktop/AML_error_recognition-main\ 2

# Check if files exist
ls core/models/er_lstm.py
ls data/features  # Should contain pre-extracted features
```

### Step 2: Train LSTM on Step Split

```bash
# Train LSTM with Omnivore features
python train_er.py \
    --variant LSTM \
    --backbone omnivore \
    --split step \
    --num_epochs 50 \
    --lr 1e-3 \
    --batch_size 1 \
    --weight_decay 1e-3 \
    --seed 42
```

**Expected Output**:
```
Training step model and testing on step level
Epoch: 1, Train Loss: X.XXXX, Test Loss: X.XXXX, F1: XX.XX, AUC: XX.XX
Epoch: 2, Train Loss: X.XXXX, Test Loss: X.XXXX, F1: XX.XX, AUC: XX.XX
...
```

### Step 3: Train LSTM on Recordings Split

```bash
# Train LSTM with Omnivore features on recordings split
python train_er.py \
    --variant LSTM \
    --backbone omnivore \
    --split recordings \
    --num_epochs 50 \
    --lr 1e-3 \
    --batch_size 1 \
    --weight_decay 1e-3 \
    --seed 42
```

### Step 4: Evaluate Trained Model

```bash
# Evaluate on step split (threshold=0.6)
python -m core.evaluate \
    --variant LSTM \
    --backbone omnivore \
    --ckpt checkpoints/error_recognition/LSTM/omnivore/error_recognition_LSTM_omnivore_step_best.pt \
    --split step \
    --threshold 0.6

# Evaluate on recordings split (threshold=0.4)
python -m core.evaluate \
    --variant LSTM \
    --backbone omnivore \
    --ckpt checkpoints/error_recognition/LSTM/omnivore/error_recognition_LSTM_omnivore_recordings_best.pt \
    --split recordings \
    --threshold 0.4
```

## Training Alternative Variants

### LSTM with Attention

```bash
python train_er.py \
    --variant LSTM_Attention \
    --backbone omnivore \
    --split step \
    --num_epochs 50
```

### GRU Baseline

```bash
python train_er.py \
    --variant GRU \
    --backbone omnivore \
    --split step \
    --num_epochs 50
```

## Training with Different Backbones

### SlowFast Features

```bash
python train_er.py \
    --variant LSTM \
    --backbone slowfast \
    --split step \
    --num_epochs 50
```

## Hyperparameter Tuning

### Learning Rate Search

```bash
# Low learning rate
python train_er.py --variant LSTM --backbone omnivore --split step --lr 1e-4

# Medium learning rate (default)
python train_er.py --variant LSTM --backbone omnivore --split step --lr 1e-3

# High learning rate
python train_er.py --variant LSTM --backbone omnivore --split step --lr 5e-3
```

### Dropout Tuning

Modify `core/models/er_lstm.py` and change dropout rates:

```python
# In ErLSTM.__init__()
self.dropout1 = nn.Dropout(0.2)  # Try: 0.0, 0.2, 0.3, 0.5
self.dropout2 = nn.Dropout(0.2)
self.dropout3 = nn.Dropout(0.1)
```

### Architecture Variants

**Single Layer LSTM**:

Modify `er_lstm.py`:
```python
# Remove lstm2 and use only lstm1
self.lstm = nn.LSTM(
    input_size=input_dim,
    hidden_size=512,
    num_layers=1,
    batch_first=True,
    bidirectional=True
)
# Update forward() accordingly
```

**Unidirectional LSTM**:

```python
# Set bidirectional=False
self.lstm1 = nn.LSTM(
    input_size=input_dim,
    hidden_size=512,
    num_layers=1,
    batch_first=True,
    bidirectional=False  # Changed
)
```

## Monitoring Training with Weights & Biases (W&B)

### Enable W&B Logging

```bash
# Install wandb
pip install wandb

# Login to wandb
wandb login

# Train with wandb enabled (default)
python train_er.py --variant LSTM --backbone omnivore --split step --enable_wandb True
```

### Disable W&B Logging

```bash
python train_er.py --variant LSTM --backbone omnivore --split step --enable_wandb False
```

## Comparing Models

### Train All Baselines

```bash
# Train MLP
python train_er.py --variant MLP --backbone omnivore --split step --num_epochs 50

# Train Transformer
python train_er.py --variant Transformer --backbone omnivore --split step --num_epochs 50

# Train LSTM
python train_er.py --variant LSTM --backbone omnivore --split step --num_epochs 50
```

### Compare Results

Results are saved in:
- `results/error_recognition/combined_results/step_True_substep_True_threshold_0.6.csv`
- `stats/error_recognition/[variant]/[backbone]/[model_name]_training_performance.txt`

```bash
# View CSV results
cat results/error_recognition/combined_results/step_True_substep_True_threshold_0.6.csv

# View training logs
cat stats/error_recognition/LSTM/omnivore/error_recognition_step_omnivore_LSTM_video_training_performance.txt
```

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Solution**: Reduce batch size (already set to 1) or use smaller hidden sizes:

```python
# In er_lstm.py, reduce hidden sizes
self.lstm1 = nn.LSTM(..., hidden_size=256)  # Was 512
self.lstm2 = nn.LSTM(..., hidden_size=128)  # Was 256
```

### Issue 2: Model Not Found Error

**Error**: `AssertionError: Model not found for variant: LSTM and backbone: omnivore`

**Solution**: Ensure `base.py` has been updated with LSTM support. Check lines 44-70.

### Issue 3: NaN Loss During Training

**Solution**:
1. Reduce learning rate: `--lr 1e-4`
2. Gradient clipping is already enabled in `base.py:204`
3. Check for NaN in input features

### Issue 4: Very Low F1 Score

**Possible causes**:
1. Incorrect threshold (try 0.4, 0.5, 0.6)
2. Model hasn't converged (train more epochs)
3. Learning rate too high/low

## Expected Results

### Step Split (with Omnivore features)

| Model | Expected F1 | Expected AUC | Training Time (50 epochs) |
|-------|------------|--------------|---------------------------|
| MLP | ~24.3 | ~75.7 | ~30 min |
| Transformer | ~55.4 | ~75.6 | ~2-3 hours |
| **LSTM** | **50-58** | **74-78** | **~1-2 hours** |

### Recordings Split (with Omnivore features)

| Model | Expected F1 | Expected AUC | Training Time (50 epochs) |
|-------|------------|--------------|---------------------------|
| MLP | ~55.4 | ~63.0 | ~30 min |
| Transformer | ~40.7 | ~62.3 | ~2-3 hours |
| **LSTM** | **45-55** | **62-66** | **~1-2 hours** |

## Analysis and Visualization

### Extract Metrics Per Error Type

The code already prints stratified metrics during evaluation:

```
===== TEST STRATIFIED STEP METRICS (vs Normal) =====
Error Type 2: Recall=0.XX, Precision=0.XX (Positives: XX)
Error Type 3: Recall=0.XX, Precision=0.XX (Positives: XX)
Error Type 4: Recall=0.XX, Precision=0.XX (Positives: XX)
Error Type 5: Recall=0.XX, Precision=0.XX (Positives: XX)
Error Type 6: Recall=0.XX, Precision=0.XX (Positives: XX)
```

**Error Type IDs**:
- 0: No Error
- 2: Preparation Error
- 3: Temperature Error
- 4: Measurement Error
- 5: Timing Error
- 6: Technique Error

### Create Comparison Visualizations

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read results CSV
df = pd.read_csv('results/error_recognition/combined_results/step_True_substep_True_threshold_0.6.csv')

# Filter for specific models
models = df[df['Variant'].isin(['MLP', 'Transformer', 'LSTM'])]

# Plot F1 comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=models, x='Variant', y='Step F1', hue='Backbone')
plt.title('Step-Level F1 Score Comparison')
plt.ylabel('F1 Score')
plt.xlabel('Model Variant')
plt.legend(title='Backbone')
plt.savefig('f1_comparison.png')
plt.show()

# Plot AUC comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=models, x='Variant', y='Step AUC', hue='Backbone')
plt.title('Step-Level AUC Comparison')
plt.ylabel('AUC')
plt.xlabel('Model Variant')
plt.legend(title='Backbone')
plt.savefig('auc_comparison.png')
plt.show()
```

## Next Steps

After training and evaluating the LSTM baseline:

1. **Compare with Existing Baselines**
   - Analyze F1, Precision, Recall, AUC for all models
   - Compare per-error-type performance

2. **Conduct Ablation Studies**
   - Single vs. two-layer LSTM
   - Unidirectional vs. bidirectional
   - Different hidden sizes
   - Different aggregation strategies

3. **Error Analysis**
   - Identify which error types LSTM handles better
   - Analyze failure cases
   - Compare confusion matrices

4. **Write Report**
   - Document methodology
   - Present results with tables and plots
   - Discuss findings and insights
   - Suggest future improvements

## Useful Commands Summary

```bash
# Train LSTM (step split)
python train_er.py --variant LSTM --backbone omnivore --split step --num_epochs 50

# Train LSTM (recordings split)
python train_er.py --variant LSTM --backbone omnivore --split recordings --num_epochs 50

# Evaluate LSTM
python -m core.evaluate --variant LSTM --backbone omnivore \
    --ckpt checkpoints/error_recognition/LSTM/omnivore/error_recognition_LSTM_omnivore_step_best.pt \
    --split step --threshold 0.6

# Compare all models (after training)
cat results/error_recognition/combined_results/step_True_substep_True_threshold_0.6.csv

# Monitor training
tail -f stats/error_recognition/LSTM/omnivore/error_recognition_step_omnivore_LSTM_video_training_performance.txt
```

## Additional Resources

1. **[NEW_BASELINE_PROPOSAL.md](NEW_BASELINE_PROPOSAL.md)** - Detailed proposal and motivation
2. **[BASELINE_COMPARISON_ANALYSIS.md](BASELINE_COMPARISON_ANALYSIS.md)** - Comprehensive comparison framework
3. **[core/models/er_lstm.py](core/models/er_lstm.py)** - Model implementation
4. **CaptainCook4D Paper** - Original baseline descriptions
5. **Project PDF** ([3_Mistake_Detection.pdf](3_Mistake_Detection.pdf)) - Task guidelines

## Contact and Support

If you encounter issues or have questions:
1. Check the project README
2. Review error messages carefully
3. Verify all file paths and dependencies
4. Join the Discord channel mentioned in the project PDF

Good luck with your LSTM baseline implementation! 🚀
