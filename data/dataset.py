"""
Data loading and preprocessing for HRR prediction.

Handles:
    - CSV file reading and cleaning
    - Z-score normalization
    - First-order difference feature augmentation
    - Sliding window sample generation
    - PyTorch DataLoader creation
"""

import os
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class HRRDataset(Dataset):
    """PyTorch Dataset for HRR prediction samples.

    Each sample contains:
        - dynamic: [seq_len, dynamic_dim] temporal features
        - static: [static_dim] static features
        - label: scalar HRR target value
    """

    def __init__(self, data_list: List[Tuple[np.ndarray, np.ndarray, float]]):
        self.data = data_list

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dyn, stat, label = self.data[idx]
        return (
            torch.from_numpy(dyn).float(),
            torch.from_numpy(stat).float(),
            torch.tensor(label, dtype=torch.float32),
        )


def normalize_features(vals: np.ndarray) -> np.ndarray:
    """Z-score normalization per feature column.

    Args:
        vals: Array of shape [N, n_features].

    Returns:
        Normalized array, same shape.
    """
    mean = vals.mean(axis=0)
    std = vals.std(axis=0)
    return (vals - mean) / (std + 1e-6)


def compute_first_order_diff(vals: np.ndarray) -> np.ndarray:
    """Compute first-order difference along time axis.

    The first row is padded by repeating the first original row
    to maintain the same sequence length.

    Args:
        vals: Array of shape [N, n_features].

    Returns:
        Difference array of shape [N, n_features].
    """
    diffs = np.diff(vals, axis=0, prepend=vals[:1])
    return diffs


def load_all_data(
    base_dir: str,
    modes_map: Dict[str, int],
    dynamic_cols: List[str],
    seq_len: int = 15,
    label_col: str = "HRR_Label",
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Load and preprocess all subject data from CSV files.

    Walks through base_dir/mode_name/subject/ to find
    {subject}_Complete_Dataset.csv files, then generates
    sliding-window samples.

    Args:
        base_dir: Root directory of the dataset.
        modes_map: Mapping from mode folder name to integer ID.
            e.g., {"HIIT": 0, "LV-HIIT": 1, "MICT": 2}
        dynamic_cols: List of dynamic feature column names.
        seq_len: Length of the input time window (default: 15).
        label_col: Name of the target column (default: "HRR_Label").

    Returns:
        List of (dynamic, static, label) tuples.
        - dynamic: [seq_len, len(dynamic_cols) * 2] (raw + diff)
        - static: [4] (age, gender, mode_id, progress)
        - label: float
    """
    tasks = []
    for mode_name, mid in modes_map.items():
        mode_path = os.path.join(base_dir, mode_name)
        if not os.path.exists(mode_path):
            continue
        for sub in os.listdir(mode_path):
            if os.path.isdir(os.path.join(mode_path, sub)):
                tasks.append((mode_name, mid, sub))

    pool: List[Tuple[np.ndarray, np.ndarray, float]] = []

    for mode_name, mid, sub in tqdm(tasks, desc=">>> Loading dataset"):
        f_path = os.path.join(base_dir, mode_name, sub, f"{sub}_Complete_Dataset.csv")
        if not os.path.exists(f_path):
            continue

        df = pd.read_csv(f_path)

        # Drop rows with missing target or dynamic features
        required_cols = dynamic_cols + [label_col]
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        if len(df) < seq_len + 2:
            continue

        # Extract and normalize dynamic features
        vals = df[dynamic_cols].values.astype(np.float32)
        vals = normalize_features(vals)

        # Augment with first-order difference
        diffs = compute_first_order_diff(vals)
        combined = np.concatenate([vals, diffs], axis=1)

        # Static base features
        if "Age" not in df.columns or "Gender" not in df.columns:
            continue
        static_base = df[["Age", "Gender"]].values[0].astype(np.float32)

        # Sliding window: predict the next step
        for i in range(seq_len, len(df) - 1):
            progress = i / len(df)
            s_feat = np.array(
                [static_base[0], static_base[1], float(mid), progress],
                dtype=np.float32,
            )
            pool.append((combined[i - seq_len : i], s_feat, df[label_col].iloc[i + 1]))

    if len(pool) == 0:
        raise RuntimeError(
            f"No valid samples found in {base_dir}. "
            "Please check the directory structure and CSV files."
        )

    print(f">>> Total samples: {len(pool)}")
    return pool


def build_dataloaders(
    data_list: List[Tuple[np.ndarray, np.ndarray, float]],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and testing DataLoaders from indices.

    Args:
        data_list: Full list of data samples.
        train_idx: Training indices from KFold split.
        test_idx: Testing indices from KFold split.
        batch_size: Batch size for both loaders (default: 64).
        num_workers: Number of DataLoader workers (default: 0).

    Returns:
        (train_loader, test_loader)
    """
    train_set = HRRDataset([data_list[i] for i in train_idx])
    test_set = HRRDataset([data_list[i] for i in test_idx])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader
