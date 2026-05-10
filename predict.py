"""
Inference script for single-sample or batch HRR prediction.

Usage:
    # Predict from preprocessed numpy arrays
    python predict.py --model_path ./outputs/fold_1/best_model.pt --dynamic x_dyn.npy --static x_stat.npy

    # Predict from CSV file
    python predict.py --model_path ./outputs/fold_1/best_model.pt --csv input.csv
"""

import argparse
from pathlib import Path
from typing import Union

import numpy as np
import torch

from models import FinalPatentModel


def parse_args():
    parser = argparse.ArgumentParser(description="HRR Prediction Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained .pt")
    parser.add_argument("--dynamic", type=str, default=None, help="Dynamic features .npy file [seq_len, 12]")
    parser.add_argument("--static", type=str, default=None, help="Static features .npy file [4]")
    parser.add_argument("--csv", type=str, default=None, help="Alternative: input CSV file")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save", type=str, default="predictions.npy", help="Output file")
    return parser.parse_args()


def load_model(model_path: str, device: torch.device) -> FinalPatentModel:
    """Load a trained model from checkpoint."""
    model = FinalPatentModel(dynamic_dim=12, static_dim=4).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict(
    model: FinalPatentModel,
    x_d: Union[np.ndarray, torch.Tensor],
    x_s: Union[np.ndarray, torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    """Run inference on a single sample or batch.

    Args:
        model: Trained model.
        x_d: Dynamic features, shape [seq_len, 12] or [batch, seq_len, 12].
        x_s: Static features, shape [4] or [batch, 4].
        device: Computation device.

    Returns:
        Predictions as numpy array.
    """
    if isinstance(x_d, np.ndarray):
        x_d = torch.from_numpy(x_d).float()
    if isinstance(x_s, np.ndarray):
        x_s = torch.from_numpy(x_s).float()

    # Add batch dimension if needed
    if x_d.dim() == 2:
        x_d = x_d.unsqueeze(0)
    if x_s.dim() == 1:
        x_s = x_s.unsqueeze(0)

    x_d, x_s = x_d.to(device), x_s.to(device)

    with torch.no_grad():
        pred = model(x_d, x_s)

    return pred.cpu().numpy()


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    )

    print(f">>> Loading model from: {args.model_path}")
    model = load_model(args.model_path, device)

    if args.dynamic and args.static:
        x_d = np.load(args.dynamic)
        x_s = np.load(args.static)
        preds = predict(model, x_d, x_s, device)
        print(f">>> Prediction: {preds}")
        np.save(args.save, preds)
        print(f">>> Saved to: {args.save}")

    elif args.csv:
        # TODO: Implement CSV preprocessing pipeline based on your data format
        raise NotImplementedError(
            "CSV inference requires implementing your preprocessing pipeline. "
            "Please preprocess to .npy first or extend this script."
        )
    else:
        raise ValueError("Please provide either (--dynamic + --static) or --csv")


if __name__ == "__main__":
    main()
