# Google Colab Setup Guide for LSTM Baseline

This guide explains how to run the LSTM baseline on Google Colab.

## Overview

You have two options:
1. **Option A**: Push LSTM files to your GitHub repo, then clone in Colab
2. **Option B**: Upload LSTM files directly to Colab or from Google Drive

## Option A: Push to GitHub (Recommended)

### Step 1: Add LSTM Files to Your GitHub Repository

You need to push these modified/new files to your GitHub repository:

**New Files**:
- `core/models/er_lstm.py`

**Modified Files**:
- `constants.py`
- `base.py`

### How to Push to GitHub:

```bash
# Navigate to your local repository
cd "/Users/francescoliaci/Desktop/AML_error_recognition-main 2"

# Check git status
git status

# Stage the new and modified files
git add core/models/er_lstm.py
git add constants.py
git add base.py

# Commit with a descriptive message
git commit -m "Add LSTM baseline implementation

- Add ErLSTM, ErLSTMWithAttention, and ErGRU models
- Update constants.py with LSTM variants
- Update base.py to support LSTM models
"

# Push to GitHub
git push origin main
# (or use 'master' if that's your default branch)
```

### Step 2: Open the Colab Notebook

1. Open Google Colab: https://colab.research.google.com
2. Upload the notebook: **colab_lstm_training.ipynb**
   - Click "File" → "Upload notebook"
   - Select `colab_lstm_training.ipynb` from your local machine

### Step 3: Run the Notebook

The notebook will:
1. Install dependencies
2. Clone your GitHub repository (with LSTM files)
3. Mount Google Drive
4. Extract features
5. Train LSTM baseline
6. Evaluate and compare with other baselines
7. Generate visualizations

## Option B: Upload Files Directly to Colab

If you don't want to push to GitHub yet, you can upload files directly to Colab.

### Step 1: Open Original Colab Notebook

Open your existing `colab_quickstart.ipynb` in Colab.

### Step 2: Upload LSTM Files

After cloning the project in the notebook, add a new cell:

```python
# Upload files from local machine
from google.colab import files
import shutil

# Upload er_lstm.py
print("Please upload er_lstm.py")
uploaded = files.upload()
shutil.move('er_lstm.py', '/content/code/core/models/er_lstm.py')

# Upload constants.py
print("Please upload constants.py")
uploaded = files.upload()
shutil.move('constants.py', '/content/code/constants.py')

# Upload base.py
print("Please upload base.py")
uploaded = files.upload()
shutil.move('base.py', '/content/code/base.py')
```

Or, if you've uploaded them to Google Drive:

```python
# Copy from Google Drive
!cp /content/drive/MyDrive/AML_DAAI_25_26/lstm_files/er_lstm.py /content/code/core/models/
!cp /content/drive/MyDrive/AML_DAAI_25_26/lstm_files/constants.py /content/code/
!cp /content/drive/MyDrive/AML_DAAI_25_26/lstm_files/base.py /content/code/
```

### Step 3: Verify Files Are in Place

```python
# Check that files exist
!ls -lh /content/code/core/models/er_lstm.py
!grep "LSTM_VARIANT" /content/code/constants.py
!grep -A 3 "LSTM_VARIANT" /content/code/base.py
```

### Step 4: Train LSTM

```python
%%bash

cd code
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

## Quick Start Commands for Colab

Once your notebook is set up, use these commands:

### Train LSTM

```bash
%%bash
cd code
python train_er.py --variant LSTM --backbone omnivore --split step --num_epochs 50
```

### Evaluate LSTM

```bash
%%bash
cd code
python -m core.evaluate --variant LSTM --backbone omnivore \
  --ckpt checkpoints/error_recognition/LSTM/omnivore/error_recognition_step_omnivore_LSTM_video_best.pt \
  --split step --threshold 0.6
```

### View Results

```python
import pandas as pd
df = pd.read_csv('code/results/error_recognition/combined_results/step_True_substep_True_threshold_0.6.csv')
print(df[df['Variant'].isin(['MLP', 'Transformer', 'LSTM'])])
```

## Files You Need in Google Drive

Make sure you have these files in your Google Drive at `/MyDrive/AML_DAAI_25_26/`:

```
AML_DAAI_25_26/
├── data/
│   ├── backbone/
│   │   ├── omnivore.zip         # Pre-extracted Omnivore features
│   │   └── slowfast.zip         # Pre-extracted SlowFast features (optional)
│   └── checkpoint/              # Existing model checkpoints (optional)
│       ├── MLP/
│       └── Transformer/
└── results/                     # Will be created to save results
    └── lstm_baseline/
```

## Troubleshooting

### Issue: "Module not found: er_lstm"

**Solution**: Make sure `er_lstm.py` is in the correct location:
```bash
!ls -lh /content/code/core/models/er_lstm.py
```

### Issue: "LSTM_VARIANT not found in constants"

**Solution**: Verify constants.py was updated:
```bash
!grep "LSTM_VARIANT" /content/code/constants.py
```

### Issue: "Model not found for variant: LSTM"

**Solution**: Check that base.py supports LSTM:
```bash
!grep -A 5 "LSTM_VARIANT" /content/code/base.py
```

### Issue: CUDA Out of Memory

**Solution**:
1. Use T4 GPU in Colab (Runtime → Change runtime type → T4 GPU)
2. Restart runtime and try again
3. If still issues, reduce hidden sizes in er_lstm.py

## Expected Training Time

On Colab with T4 GPU:

| Split | Epochs | Expected Time |
|-------|--------|---------------|
| Step | 50 | 1-2 hours |
| Recordings | 50 | 1-2 hours |

## Saving Results

The notebook automatically saves results to Google Drive:

```python
# Results location in Drive
/MyDrive/AML_DAAI_25_26/results/lstm_baseline/
├── checkpoints/        # Trained model weights
├── results/           # CSV files with metrics
├── plots/             # Visualization plots
└── training_logs/     # Training statistics
```

## Next Steps

After training completes:

1. **Download results from Drive** for your report
2. **Compare metrics** with MLP and Transformer
3. **Analyze per-error-type performance** (printed during evaluation)
4. **Generate plots** using the visualization cells
5. **Write your report** using the documentation files

## Useful Colab Tips

### Check GPU

```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Monitor Memory

```python
!nvidia-smi
```

### Keep Session Alive

Colab sessions timeout after ~90 minutes of inactivity. To keep it alive:
1. Use Colab Pro (recommended for long training)
2. Or run a keep-alive script (not recommended, violates ToS)

### Save Checkpoints Frequently

The notebook saves checkpoints every epoch, so if Colab disconnects, you can resume training.

## Support

If you encounter issues:

1. Check the [LSTM_QUICKSTART_GUIDE.md](LSTM_QUICKSTART_GUIDE.md) for detailed instructions
2. Review the troubleshooting section above
3. Ensure all files are in the correct locations
4. Check that you have sufficient GPU memory

## Summary

**Recommended Workflow**:

1. ✅ Push LSTM files to your GitHub repository
2. ✅ Open `colab_lstm_training.ipynb` in Google Colab
3. ✅ Mount Google Drive with your data
4. ✅ Run training cells (1-2 hours per split)
5. ✅ Evaluate and compare with baselines
6. ✅ Generate visualizations
7. ✅ Download results from Drive
8. ✅ Write your report

**Total Time**: ~3-4 hours for complete training and evaluation

Good luck! 🚀
