"""
Training script for HRR prediction model with 5-Fold Cross-Validation.

Usage:
    python train.py --data_dir /path/to/data --output_dir ./outputs
"""

import os
import argparse
import json
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold

from models import FinalPatentModel
from data import load_all_data, build_dataloaders
from utils import regression_metrics, setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train HRR Prediction Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root path")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_len", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_labels(pool: list) -> Tuple[list, float, float]:
    """Z-score normalize labels and return stats for inverse transform."""
    all_labels = np.array([label for (_, _, label) in pool], dtype=np.float32)
    label_mean = float(all_labels.mean())
    label_std = float(all_labels.std()) + 1e-6
    
    normalized_pool = [
        (dyn, stat, (label - label_mean) / label_std)
        for dyn, stat, label in pool
    ]
    return normalized_pool, label_mean, label_std


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    for x_d, x_s, y in dataloader:
        x_d, x_s, y = x_d.to(device), x_s.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x_d, x_s)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        total_samples += len(y)
    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, dataloader, device, label_mean=0.0, label_std=1.0):
    """Evaluate and return denormalized truths and preds."""
    model.eval()
    preds, truths = [], []
    for x_d, x_s, y in dataloader:
        out = model(x_d.to(device), x_s.to(device))
        preds.extend(out.cpu().numpy())
        truths.extend(y.numpy())
    
    # Denormalize back to original scale
    truths = np.array(truths) * label_std + label_mean
    preds = np.array(preds) * label_std + label_mean
    return truths, preds


def train_fold(model, train_loader, test_loader, device, args, logger, fold,
               label_mean=0.0, label_std=1.0):
    """Train a single fold with early stopping."""
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.MSELoss()

    best_r2 = -float('inf')
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        truths, preds = evaluate(model, test_loader, device, label_mean, label_std)
        metrics = regression_metrics(truths, preds)

        history.append({"epoch": epoch, "loss": train_loss, **metrics})
        scheduler.step(metrics['r2'])

        if metrics['r2'] > best_r2:
            best_r2 = metrics['r2']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        logger.info(
            f"  [Fold {fold}] Epoch {epoch:02d}/{args.epochs} | "
            f"Loss={train_loss:.4f} | R2={metrics['r2']:.4f} | "
            f"MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.2e}"
        )

        if patience_counter >= args.patience:
            logger.info(f"  >> Early stopping at epoch {epoch}")
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save
    fold_dir = os.path.join(args.output_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    model_path = os.path.join(fold_dir, "best_model.pt")
    torch.save(best_state, model_path)

    history_path = os.path.join(fold_dir, "history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    truths, preds = evaluate(model, test_loader, device, label_mean, label_std)
    final_metrics = regression_metrics(truths, preds)

    logger.info(f">>> Fold {fold} best R2: {best_r2:.4f}")
    return {
        "fold": fold,
        "metrics": final_metrics,
        "model_path": model_path,
        "truths": truths,
        "preds": preds,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log_file = os.path.join(args.output_dir, "train.log")
    logger = setup_logger(name="hrr_train", log_file=log_file)
    logger.info("=" * 60)
    logger.info("HRR Prediction Model Training")
    logger.info("=" * 60)
    logger.info(f"Arguments: {json.dumps(vars(args), indent=2)}")

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    logger.info(f"Using device: {device}")
    set_seed(args.seed)

    MODES = {"HIIT": 0, "LV-HIIT": 1, "MICT": 2}
    DYN_COLS = ["Global_SampEn", "Global_RMS", "Mag_Var", "V1_range", "V2_std", "V3_range"]

    # Load data
    logger.info(f">>> Loading data from: {args.data_dir}")
    pool = load_all_data(base_dir=args.data_dir, modes_map=MODES, dynamic_cols=DYN_COLS, seq_len=args.seq_len)
    logger.info(f">>> Raw samples: {len(pool)}")

    # Normalize labels (Z-score)
    pool, label_mean, label_std = normalize_labels(pool)
    logger.info(f">>> Label stats: mean={label_mean:.4f}, std={label_std:.4f}")

    # Save label stats
    np.savez(os.path.join(args.output_dir, "label_stats.npz"), mean=label_mean, std=label_std)

    # Cross-validation
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    global_truths, global_preds = [], []
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(pool), start=1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Fold {fold}/{args.n_splits}")
        logger.info(f"{'='*50}")

        train_loader, test_loader = build_dataloaders(
            pool, train_idx, test_idx, batch_size=args.batch_size, num_workers=args.num_workers
        )

        model = FinalPatentModel(
            dynamic_dim=len(DYN_COLS) * 2,
            static_dim=4,
            d_model=args.d_model,
            nhead=args.nhead,
            num_transformer_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)

        result = train_fold(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            args=args,
            logger=logger,
            fold=fold,
            label_mean=label_mean,
            label_std=label_std,
        )

        fold_results.append(result)
        global_truths.extend(result["truths"])
        global_preds.extend(result["preds"])

    # Global metrics (on original scale)
    global_truths = np.array(global_truths)
    global_preds = np.array(global_preds)
    global_metrics = regression_metrics(global_truths, global_preds)

    logger.info(f"\n{'='*60}")
    logger.info("Cross-Validation Complete (Original Scale)")
    logger.info(f"{'='*60}")
    logger.info(f"Global R2   = {global_metrics['r2']:.4f}")
    logger.info(f"Global MAE  = {global_metrics['mae']:.4f}")
    logger.info(f"Global RMSE = {global_metrics['rmse']:.4f}")
    logger.info(f"Global MAPE = {global_metrics['mape']:.2f}%")

    # Save
    results_summary = {
        "args": vars(args),
        "label_mean": label_mean,
        "label_std": label_std,
        "global_metrics": global_metrics,
        "fold_metrics": [r["metrics"] for r in fold_results],
    }
    results_path = os.path.join(args.output_dir, "results_summary.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
