#!/usr/bin/env python3

import os
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from model import ConvLSTM

# === User Settings ===
DATA_FILE    = "filled_smoothed_data.nc"
INPUT_VARS   = [
    "pr", "rmax", "rmin", "sph", "srad",
    "th", "tmmn", "tmmx", "vs", "erc",
    "eto", "bi", "fm100", "fm1000", "etr",
    "vpd", "LAI_AVE"
]
TARGET_VAR   = "frp"
SEQ_LENGTH   = 24         # number of time steps per input sequence
BATCH_SIZE   = 2          # limited to 2 from 8 due to hardware
EPOCHS       = 20
LR           = 1e-3
HIDDEN_DIM   = 32         # limited to 32 from 64 due to hardware
NUM_LAYERS   = 2
KERNEL_SIZE  = (3, 3)
TRAIN_SPLIT  = 0.8        # fraction of sequences for training
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =======================

def prepare_data():
    ds = xr.open_dataset(DATA_FILE)
    # stack environmental inputs into (time, H, W, C)
    X_arr = np.stack([ds[var].values for var in INPUT_VARS], axis=-1)
    y_arr = ds[TARGET_VAR].values  # (time, H, W)

    n_time, H, W, C = X_arr.shape

    # normalize per-channel
    scaler = StandardScaler()
    X_flat = X_arr.reshape(-1, C)
    X_norm = scaler.fit_transform(X_flat).reshape(n_time, H, W, C)

    # build sequences
    n_samples = n_time - SEQ_LENGTH
    X_seq = np.array([X_norm[i : i + SEQ_LENGTH] for i in range(n_samples)])
    y_seq = np.array([y_arr[i + SEQ_LENGTH] for i in range(n_samples)])

    # split train/val
    split = int(TRAIN_SPLIT * n_samples)
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]

    # to torch Tensors, reorder to (B, T, C, H, W); target (B, 1, H, W)
    def to_tensor(X, y):
        X_t = torch.from_numpy(X).float().permute(0, 1, 4, 2, 3)
        y_t = torch.from_numpy(y).float().unsqueeze(1)
        return X_t, y_t

    X_tr, y_tr = to_tensor(X_train, y_train)
    X_va, y_va = to_tensor(X_val,   y_val)

    train_ds = TensorDataset(X_tr, y_tr)
    val_ds   = TensorDataset(X_va, y_va)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    return train_loader, val_loader, C, H, W

def build_model(input_dim):
    model = ConvLSTM(
        input_dim=input_dim,
        hidden_dim=[HIDDEN_DIM]*NUM_LAYERS,
        kernel_size=KERNEL_SIZE,
        num_layers=NUM_LAYERS,
        batch_first=True,
        bias=True,
        return_all_layers=False
    )
    # final conv to map hidden_dim → 1 channel (FRP)
    model.final_conv = nn.Conv2d(HIDDEN_DIM, 1, kernel_size=3, padding=1)
    return model.to(DEVICE)

def train():
    train_loader, val_loader, C, H, W = prepare_data()
    model = build_model(C)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS+1):
        # MARK: Training
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            # forward through ConvLSTM
            layer_outs, _ = model(Xb)  # list of length 1
            # take last timestep output: shape (B, seq, HIDDEN_DIM, H, W)
            h_seq = layer_outs[0][:, -1]  # (B, HIDDEN_DIM, H, W)
            preds = model.final_conv(h_seq)  # (B, 1, H, W)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)
        train_loss /= len(train_loader.dataset)

        # MARK: Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                layer_outs, _ = model(Xb)
                h_seq = layer_outs[0][:, -1]
                preds = model.final_conv(h_seq)
                val_loss += criterion(preds, yb).item() * Xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch}/{EPOCHS} — "
              f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

    # checkpoint
    ckpt_path = "convlstm_fire.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model weights saved to {ckpt_path}")

if __name__ == "__main__":
    train()
