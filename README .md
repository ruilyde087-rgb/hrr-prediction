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

## K2P Twin: Kinematics to Physiology Digital Twin Model

This repository implements the HRR prediction module integrated with the **Kinematics to Physiology Digital Twin Model (K2P Twin)**.

## Sample Data

A de-identified sample dataset from one participant (M01) is provided for demonstration purposes.

| File | Description |
|------|-------------|
| `M01_IMU_PatentFeatures.csv` | Filtered and processed IMU sensor data with patent features from participant M01 |
| `hr_M01.csv` | Filtered heart rate belt recordings from participant M01 |

> **Note**: Both datasets have been filtered and anonymized. All personally identifiable information has been removed.

## K2P Twin HIIT Protocol Demonstrations

The following videos demonstrate the **Kinematics to Physiology Digital Twin Model (K2P Twin)** platform interface during a **HIIT protocol**, presenting three key phases of the exercise intervention:

<table>
  <tr>
    <td align="center">
      <video src="./demos/Train.mp4" width="100%" controls></video><br/>
      <b>Train (Sprint Bout)</b><br/>
      <sub>First high-intensity sprint phase of HIIT</sub>
    </td>
    <td align="center">
      <video src="./demos/Train%20(Interval).mp4" width="100%" controls></video><br/>
      <b>Train (Interval)</b><br/>
      <sub>Inter-bout recovery interval between sprint bouts</sub>
    </td>
    <td align="center">
      <video src="./demos/Cooldown.mp4" width="100%" controls></video><br/>
      <b>Cooldown</b><br/>
      <sub>Post-exercise cool-down recovery phase</sub>
    </td>
  </tr>
</table>

## Ethics Statement

The experiment protocol was approved by the Academic Ethics Committee of Sichuan Agricultural University (Approval No: H20250017) and conducted in accordance with the Declaration of Helsinki. All participants provided written informed consent prior to participation.

## Contact

For questions, collaboration requests, or data inquiries, please contact:
📧 **Rui Deng** - dengrui1@stu.sicau.edu.cn

## License

MIT License
## Contact

For questions, collaboration requests, or data inquiries, please contact:
📧 **Rui Deng** - dengrui1@stu.sicau.edu.cn

## License

MIT License
