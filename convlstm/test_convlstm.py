#!/usr/bin/env python3
"""
test_convlstm.py

Load the trained ConvLSTM model and evaluate it on the held-out test set,
using a higher classification threshold for better precision.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from train_convlstm import prepare_data, build_model, DEVICE

# classification threshold
THRESHOLD = 0.65

def test():
    # only unpack the test_loader
    _, _, test_loader, C = prepare_data()

    # rebuild model & load weights
    model = build_model(C)
    model.load_state_dict(torch.load("convlstm_fire.pth",
                                     map_location=DEVICE))
    model.to(DEVICE).eval()

    # accumulators for regression
    mse_sum = 0.0
    mae_sum = 0.0
    n_pixels = 0

    all_probs = []
    all_truth = []

    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)

            # forward through ConvLSTM
            layer_outs, _ = model(Xb)
            h_seq = layer_outs[0][:, -1]           # (B, HIDDEN_DIM, H, W)

            # regression output
            frp_hat = model.reg_head(h_seq)        # (B,1,H,W)

            # classification output
            logits  = model.clf_head(h_seq)        # (B,1,H,W)
            probs   = torch.sigmoid(logits)        # (B,1,H,W)

            # regression metrics
            mse_sum += nn.MSELoss(reduction="sum")(frp_hat, yb).item()
            mae_sum += nn.L1Loss(reduction="sum")(frp_hat, yb).item()
            n_pixels += frp_hat.numel()

            # collect for classification
            all_probs.append(probs.cpu().numpy().ravel())
            all_truth.append(yb.cpu().numpy().ravel())

    # finalize regression scores
    mse  = mse_sum  / n_pixels
    mae  = mae_sum  / n_pixels
    rmse = np.sqrt(mse)
    print(f"Test Regression → MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}")

    # flatten arrays
    probs_flat = np.concatenate(all_probs)
    truth_flat = np.concatenate(all_truth)

    # classification: threshold on P(fire)
    pred_fire = (probs_flat > THRESHOLD).astype(int)
    true_fire = (truth_flat >  0.0).astype(int)

    acc  = accuracy_score(true_fire, pred_fire)
    prec = precision_score(true_fire, pred_fire, zero_division=0)
    rec  = recall_score(true_fire, pred_fire, zero_division=0)
    f1   = f1_score(true_fire, pred_fire, zero_division=0)

    print(f"Test Classification (@{THRESHOLD:.2f}) → "
          f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, "
          f"Recall: {rec:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    test()
