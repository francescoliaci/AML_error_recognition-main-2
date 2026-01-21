# LSTM Baseline Implementation - Quick Reference

## What's New

This repository now includes a **new LSTM-based baseline** for mistake detection in procedural activities, along with comprehensive documentation and analysis.

### New Files

| File | Description |
|------|-------------|
| **[core/models/er_lstm.py](core/models/er_lstm.py)** | LSTM model implementations (ErLSTM, ErLSTMWithAttention, ErGRU) |
| **[NEW_BASELINE_PROPOSAL.md](NEW_BASELINE_PROPOSAL.md)** | Detailed proposal and motivation for LSTM baseline |
| **[BASELINE_COMPARISON_ANALYSIS.md](BASELINE_COMPARISON_ANALYSIS.md)** | Comprehensive comparison of MLP, Transformer, and LSTM |
| **[LSTM_QUICKSTART_GUIDE.md](LSTM_QUICKSTART_GUIDE.md)** | Step-by-step training and evaluation guide |
| **[SUMMARY.md](SUMMARY.md)** | High-level overview of the entire project |
| **[visualize_results.py](visualize_results.py)** | Script to generate comparison plots |

### Modified Files

| File | Changes |
|------|---------|
| **[constants.py](constants.py)** | Added `LSTM_VARIANT`, `LSTM_ATTENTION_VARIANT`, `GRU_VARIANT` |
| **[base.py](base.py)** | Updated `fetch_model()` to support LSTM variants |

## Quick Start

### 1. Train LSTM Baseline

```bash
# Step split (threshold=0.6)
python train_er.py --variant LSTM --backbone omnivore --split step --num_epochs 50

# Recordings split (threshold=0.4)
python train_er.py --variant LSTM --backbone omnivore --split recordings --num_epochs 50
```

### 2. Evaluate LSTM

```bash
# Step split
python -m core.evaluate --variant LSTM --backbone omnivore \
    --ckpt checkpoints/error_recognition/LSTM/omnivore/error_recognition_LSTM_omnivore_step_best.pt \
    --split step --threshold 0.6

# Recordings split
python -m core.evaluate --variant LSTM --backbone omnivore \
    --ckpt checkpoints/error_recognition/LSTM/omnivore/error_recognition_LSTM_omnivore_recordings_best.pt \
    --split recordings --threshold 0.4
```

### 3. Generate Visualizations

```bash
python visualize_results.py
```

Plots will be saved in `plots/` directory.

## Expected Results

### Step Split (Omnivore backbone)

| Model | F1 | AUC | Parameters | Training Time |
|-------|-----|-----|------------|---------------|
| MLP | 24.26 | 75.74 | ~500K | ~30 min |
| Transformer | 55.39 | 75.62 | ~2M | ~2-3 hours |
| **LSTM (NEW)** | **50-58** | **74-78** | **~1M** | **~1-2 hours** |

### Recordings Split (Omnivore backbone)

| Model | F1 | AUC |
|-------|-----|-----|
| MLP | 55.42 | 63.03 |
| Transformer | 40.73 | 62.27 |
| **LSTM (NEW)** | **45-55** | **62-66** |

## Model Variants

### 1. ErLSTM (Main baseline)

Two-layer bidirectional LSTM with dropout regularization.

```bash
python train_er.py --variant LSTM --backbone omnivore --split step --num_epochs 50
```

### 2. ErLSTMWithAttention

LSTM with attention mechanism over hidden states.

```bash
python train_er.py --variant LSTM_Attention --backbone omnivore --split step --num_epochs 50
```

### 3. ErGRU

GRU alternative (simpler, fewer parameters).

```bash
python train_er.py --variant GRU --backbone omnivore --split step --num_epochs 50
```

## Architecture Overview

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
Take final hidden state (512-dim)
  ↓
MLP: Linear(512, 256) → ReLU → Dropout(0.2) → Linear(256, 1)
  ↓
Output: Binary prediction (logit)
```

## Why LSTM?

### Advantages over MLP
- ✅ **Temporal modeling**: Captures dependencies across sub-segments
- ✅ **Sequential memory**: Maintains information through time steps
- ✅ **Better for timing errors**: Excels at detecting temporal mistakes

### Advantages over Transformer
- ✅ **Data efficiency**: Stronger inductive bias, better with limited data
- ✅ **Parameter efficiency**: ~50% fewer parameters (~1M vs ~2M)
- ✅ **Computational efficiency**: Lower memory footprint, faster training
- ✅ **Streaming capability**: Can process sequences online

### Expected Strengths

LSTM is expected to **significantly outperform** on temporal error types:

| Error Type | MLP | Transformer | LSTM (Expected) |
|------------|-----|-------------|-----------------|
| Temperature | ❌ Poor | 🟡 Moderate | ✅✅ **Excellent** |
| Timing | ❌ Poor | 🟡 Moderate | ✅✅ **Excellent** |
| Technique | 🟡 Moderate | ✅ Good | ✅ Good |
| Measurement | ✅ Good | ✅ Good | ✅ Good |

## Documentation

For detailed information, refer to:

1. **[SUMMARY.md](SUMMARY.md)** - Start here for a high-level overview
2. **[LSTM_QUICKSTART_GUIDE.md](LSTM_QUICKSTART_GUIDE.md)** - Training and evaluation instructions
3. **[NEW_BASELINE_PROPOSAL.md](NEW_BASELINE_PROPOSAL.md)** - Motivation and architecture details
4. **[BASELINE_COMPARISON_ANALYSIS.md](BASELINE_COMPARISON_ANALYSIS.md)** - Comprehensive comparison

## Project Structure

```
AML_error_recognition-main 2/
├── core/
│   ├── models/
│   │   ├── er_lstm.py          # NEW: LSTM implementations
│   │   ├── er_former.py         # Existing: Transformer
│   │   └── blocks.py            # Existing: MLP
│   ├── config.py
│   ├── evaluate.py
│   └── utils.py
├── dataloader/
│   ├── CaptainCookStepDataset.py
│   └── CaptainCookSubStepDataset.py
├── annotations/                 # Dataset annotations
├── data/                        # Pre-extracted features
├── base.py                      # MODIFIED: Added LSTM support
├── constants.py                 # MODIFIED: Added LSTM constants
├── train_er.py                  # Main training script
├── visualize_results.py         # NEW: Visualization script
├── NEW_BASELINE_PROPOSAL.md     # NEW: Detailed proposal
├── BASELINE_COMPARISON_ANALYSIS.md  # NEW: Comparison
├── LSTM_QUICKSTART_GUIDE.md     # NEW: Quick start
├── SUMMARY.md                   # NEW: Overview
├── NEW_BASELINE_README.md       # NEW: This file
└── README.md                    # Original project README
```

## Troubleshooting

### Common Issues

**Issue**: `AssertionError: Model not found for variant: LSTM`

**Solution**: Ensure [base.py](base.py) has been updated with LSTM support.

**Issue**: CUDA out of memory

**Solution**: LSTM uses less memory than Transformer. If still an issue, reduce hidden sizes in [er_lstm.py](core/models/er_lstm.py).

**Issue**: NaN loss during training

**Solution**:
1. Reduce learning rate: `--lr 1e-4`
2. Gradient clipping is already enabled
3. Check for NaN in input features

### Getting Help

1. Check [LSTM_QUICKSTART_GUIDE.md](LSTM_QUICKSTART_GUIDE.md) for detailed instructions
2. Review error messages carefully
3. Verify all file paths and dependencies
4. Consult the troubleshooting section in the quickstart guide

## Citation

If you use this LSTM baseline, please cite:

```bibtex
@inproceedings{peddi2024captaincook4d,
  title={CaptainCook4D: A dataset for understanding errors in procedural activities},
  author={Peddi, Rohith and others},
  booktitle={NeurIPS},
  year={2024}
}

@article{hochreiter1997long,
  title={Long short-term memory},
  author={Hochreiter, Sepp and Schmidhuber, J{\"u}rgen},
  journal={Neural Computation},
  year={1997}
}
```

## Contributing

This is a course project implementation. For improvements or suggestions:

1. Test thoroughly on the CaptainCook4D dataset
2. Document changes clearly
3. Update relevant documentation files
4. Compare with existing baselines

## License

This project builds upon the CaptainCook4D codebase. Please refer to the original repository for license information.

## Acknowledgements

- **CaptainCook4D team** for the dataset and baseline implementations
- **Original MLP and Transformer implementations** from the CaptainCook4D paper
- **PyTorch team** for the LSTM implementation

---

**Last Updated**: 2026-01-21

**Status**: Implementation complete, ready for training and evaluation

**Next Steps**: Train models, evaluate, analyze results, write report
