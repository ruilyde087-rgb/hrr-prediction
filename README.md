# Heart Rate Recovery (HRR) Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A deep learning model for predicting Heart Rate Recovery (HRR) using a hybrid architecture of **Bi-LSTM**, **Transformer**, and **Cross-Attention**.

## Architecture

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
│                MLP Regressor (linear output)
│                        ↓
│              Predicted Normalized HRR
```

## Project Structure

```
.
├── config.yaml              # Hyperparameter configuration
├── train.py                 # 5-Fold Cross-Validation training
├── evaluate.py              # Model evaluation
├── predict.py               # Inference script
├── requirements.txt         # Python dependencies
├── models/
│   ├── __init__.py
│   └── hrr_model.py         # Bi-LSTM + Transformer model
├── data/
│   ├── __init__.py
│   └── dataset.py           # Data loading & preprocessing
└── utils/
    ├── __init__.py
    ├── metrics.py           # Regression metrics
    └── logger.py            # Logging utilities
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Format

```
Final_Combined_Dataset/
├── HIIT/
│   ├── Subject_01/
│   │   └── Subject_01_Complete_Dataset.csv
│   └── ...
├── LV-HIIT/
│   └── ...
└── MICT/
    └── ...
```

**Required CSV columns**: `HRR_Label`, `Age`, `Gender`, `Global_SampEn`, `Global_RMS`, `Mag_Var`, `V1_range`, `V2_std`, `V3_range`

## Usage

### Training

```bash
python train.py --data_dir "/path/to/Final_Combined_Dataset" --epochs 50
```

### Evaluation

```bash
python evaluate.py --model_path ./outputs/fold_1/best_model.pt --data_dir "/path/to/data"
```

### Inference

```bash
python predict.py --model_path ./outputs/fold_1/best_model.pt --dynamic x_dyn.npy --static x_stat.npy
```

## Key Features

- **Label normalization**: Z-score normalization for stable training
- **Early stopping**: Prevents overfitting with configurable patience
- **Cross-validation**: 5-Fold CV for robust evaluation
- **Attention weights**: Extract attention maps for interpretability

## License

MIT License
