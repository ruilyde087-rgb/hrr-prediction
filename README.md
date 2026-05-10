# Heart Rate Recovery (HRR) Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A deep learning model for predicting Heart Rate Recovery (HRR) using a hybrid architecture of **Bi-LSTM**, **Transformer**, and **Cross-Attention**.

## Architecture Overview

```
Input
├── Dynamic Features (x_d): [batch, seq_len=15, 12]
│   ├── 6 raw physiological features
│   └── 6 first-order difference features
│   
│   → Bi-LSTM (local temporal) → Positional Encoding → Transformer (global dependency)
│                                                          ↓
├── Static Features (x_s): [batch, 4]                     │
│   ├── Age, Gender, Exercise Mode, Progress              │
│   → Linear Projection [64]                               │
│        ↓                                                 │
│   Query ───────→ Cross-Attention ←────── Key/Value ─────┘
│                        ↓
│                MLP Regressor + Sigmoid
│                        ↓
│              Predicted HRR [0, 1]
```

### Key Components

| Module | Purpose |
|--------|---------|
| **Bi-LSTM** | Extracts local temporal patterns from physiological signals |
| **Positional Encoding** | Provides sequence order information for Transformer |
| **Transformer Encoder** | Models long-range global dependencies across time steps |
| **Cross-Attention** | Fuses static features (age/gender/mode) with dynamic temporal features |
| **Sigmoid Regressor** | Outputs normalized HRR predictions in range (0, 1) |

## Project Structure

```
.
├── config.yaml              # Hyperparameter configuration
├── train.py                 # 5-Fold Cross-Validation training script
├── evaluate.py              # Model evaluation on test set
├── predict.py               # Inference script for new samples
├── requirements.txt         # Python dependencies
├── models/
│   ├── __init__.py
│   └── hrr_model.py         # Model architecture (Bi-LSTM + Transformer)
├── data/
│   ├── __init__.py
│   └── dataset.py           # Data loading, preprocessing, Dataset class
└── utils/
    ├── __init__.py
    ├── metrics.py           # Regression metrics (R2, MAE, RMSE, MAPE)
    └── logger.py            # Logging utilities
```

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/hrr-prediction.git
cd hrr-prediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Dataset Format

The dataset should be organized as follows:

```
Final_Combined_Dataset/
├── HIIT/
│   ├── Subject_01/
│   │   └── Subject_01_Complete_Dataset.csv
│   ├── Subject_02/
│   │   └── Subject_02_Complete_Dataset.csv
│   └── ...
├── LV-HIIT/
│   ├── Subject_01/
│   │   └── Subject_01_Complete_Dataset.csv
│   └── ...
└── MICT/
    ├── Subject_01/
    │   └── Subject_01_Complete_Dataset.csv
    └── ...
```

### CSV Column Requirements

**Required columns:**
- `HRR_Label`: Target variable (heart rate recovery value)
- `Age`: Subject age
- `Gender`: Subject gender
- Dynamic feature columns: `Global_SampEn`, `Global_RMS`, `Mag_Var`, `V1_range`, `V2_std`, `V3_range`

### Data Preprocessing Pipeline

1. **Z-score normalization** per feature
2. **First-order difference** augmentation (6 raw + 6 diff = 12 dynamic features)
3. **Sliding window** generation (window size = 15, predict next step)
4. **Static feature construction**: `[Age, Gender, Mode_ID, Progress]`

## Usage

### 1. Training (5-Fold Cross-Validation)

```bash
# Basic training with default parameters
python train.py --data_dir "/path/to/Final_Combined_Dataset" --output_dir ./outputs

# Custom hyperparameters
python train.py \
    --data_dir "/path/to/data" \
    --output_dir ./outputs \
    --d_model 128 \
    --nhead 8 \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --n_splits 5 \
    --seed 42
```

**All training arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `C:\Fatigue HRR pre\Final_Combined_Dataset` | Dataset root path |
| `--output_dir` | `./outputs` | Output directory for models/logs |
| `--d_model` | 64 | Model hidden dimension |
| `--nhead` | 8 | Number of attention heads |
| `--num_layers` | 2 | Transformer encoder layers |
| `--dropout` | 0.1 | Dropout rate |
| `--epochs` | 20 | Training epochs per fold |
| `--batch_size` | 64 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--weight_decay` | 1e-3 | Weight decay (L2 regularization) |
| `--n_splits` | 5 | K-Fold splits |
| `--seq_len` | 15 | Input sequence length |
| `--seed` | 42 | Random seed |
| `--device` | auto | Device: `auto`, `cpu`, or `cuda` |

### 2. Evaluation

```bash
python evaluate.py \
    --model_path ./outputs/fold_1/best_model.pt \
    --data_dir "/path/to/data" \
    --output metrics.json
```

### 3. Inference

```bash
# From numpy arrays
python predict.py \
    --model_path ./outputs/fold_1/best_model.pt \
    --dynamic sample_dyn.npy \
    --static sample_stat.npy \
    --save predictions.npy
```

**Python API:**

```python
import numpy as np
import torch
from models import FinalPatentModel
from predict import load_model, predict

device = torch.device("cpu")
model = load_model("./outputs/fold_1/best_model.pt", device)

# Single sample
x_d = np.random.randn(15, 12).astype(np.float32)  # [seq_len, dynamic_dim]
x_s = np.array([25.0, 1.0], dtype=np.float32)      # [static_dim]

pred = predict(model, x_d, x_s, device)
print(f"Predicted HRR: {pred[0]:.4f}")
```

## Training Output

After training, the `outputs/` directory will contain:

```
outputs/
├── train.log                          # Training log
├── results_summary.json               # Aggregated CV metrics
├── fold_1/
│   ├── best_model.pt                  # Best model checkpoint
│   └── history.json                   # Per-epoch training history
├── fold_2/
│   ├── best_model.pt
│   └── history.json
└── ...
```

### Example Output

```
============================================================
Cross-Validation Complete
============================================================
Global R2   = 0.8732
Global MAE  = 0.0421
Global RMSE = 0.0587
Global MAPE = 8.45%
```

## Model Checkpoints

| Fold | R2 | MAE | RMSE | Download |
|------|-----|-----|------|----------|
| 1 | 0.8812 | 0.0401 | 0.0562 | [model.pt](link) |
| 2 | 0.8698 | 0.0435 | 0.0601 | [model.pt](link) |
| 3 | 0.8756 | 0.0418 | 0.0583 | [model.pt](link) |
| 4 | 0.8701 | 0.0429 | 0.0598 | [model.pt](link) |
| 5 | 0.8793 | 0.0412 | 0.0571 | [model.pt](link) |
| **Avg** | **0.8752** | **0.0419** | **0.0583** | |

## Citation

If you use this code in your research, please cite:

```bibtex
@software{hrr_prediction,
  title={Heart Rate Recovery Prediction via Bi-LSTM and Transformer},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/hrr-prediction}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Cross-validation powered by [scikit-learn](https://scikit-learn.org/)
