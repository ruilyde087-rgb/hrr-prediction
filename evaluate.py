"""
Evaluation script to assess a trained model on test data.

Usage:
    python evaluate.py --model_path ./outputs/fold_1/best_model.pt --data_dir /path/to/data
"""

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from models import FinalPatentModel
from data import load_all_data, build_dataloaders
from utils import regression_metrics, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HRR Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pt model")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=15)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path")
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    preds, truths = [], []
    for x_d, x_s, y in dataloader:
        out = model(x_d.to(device), x_s.to(device))
        preds.extend(out.cpu().numpy())
        truths.extend(y.numpy())
    return np.array(truths), np.array(preds)


def main():
    args = parse_args()
    logger = setup_logger()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    )

    MODES = {"HIIT": 0, "LV-HIIT": 1, "MICT": 2}
    DYN_COLS = ["Global_SampEn", "Global_RMS", "Mag_Var", "V1_range", "V2_std", "V3_range"]

    logger.info(f">>> Loading data from: {args.data_dir}")
    pool = load_all_data(args.data_dir, MODES, DYN_COLS, seq_len=args.seq_len)

    # Use last 20% as test set (or modify as needed)
    n = len(pool)
    test_idx = np.arange(int(n * 0.8), n)
    train_idx = np.arange(0, int(n * 0.8))
    _, test_loader = build_dataloaders(pool, train_idx, test_idx, batch_size=args.batch_size)

    logger.info(f">>> Loading model from: {args.model_path}")
    model = FinalPatentModel(dynamic_dim=12, static_dim=4).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)

    truths, preds = evaluate_model(model, test_loader, device)
    metrics = regression_metrics(truths, preds)

    logger.info("=" * 40)
    logger.info(f"R2   = {metrics['r2']:.4f}")
    logger.info(f"MAE  = {metrics['mae']:.4f}")
    logger.info(f"RMSE = {metrics['rmse']:.4f}")
    logger.info(f"MAPE = {metrics['mape']:.2f}%")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to: {args.output}")


if __name__ == "__main__":
    main()
