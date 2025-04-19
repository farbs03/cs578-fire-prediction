#!/usr/bin/env python3
"""
train_convlstm.py

Train a ConvLSTM with a two-head (classification+regression) loss,
clamping class weights and rebalancing regression emphasis so the
model learns to detect rare fires more effectively.
"""

import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import ConvLSTM

# === User Settings ===
DATA_FILE        = "filled_smoothed_data.nc"
INPUT_VARS       = [
    "pr","rmax","rmin","sph","srad","th",
    "tmmn","tmmx","vs","erc","eto","bi",
    "fm100","fm1000","etr","vpd","LAI_AVE"
]
TARGET_VAR       = "frp"
SEQ_LENGTH       = 24
BATCH_SIZE       = 2
EPOCHS           = 20
LR               = 1e-3
HIDDEN_DIM       = 32
NUM_LAYERS       = 2
KERNEL_SIZE      = (3, 3)
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Two‐head loss settings
USE_TWO_HEAD     = True
REG_LOSS_WEIGHT  = 1.0    # lower emphasis on regression so BCE dominates
OVERSAMPLE_FIRE  = True
MAX_POS_WEIGHT   = 10.0   # clamp computed pos_weight here
# =======================

def prepare_data():
    ds    = xr.open_dataset(DATA_FILE)
    X_arr = np.stack([ds[v].values for v in INPUT_VARS], axis=-1)  # (T,H,W,C)
    y_arr = ds[TARGET_VAR].values                                  # (T,H,W)
    T, H, W, C = X_arr.shape

    scaler = StandardScaler()
    X_flat = X_arr.reshape(-1, C)
    X_norm = scaler.fit_transform(X_flat).reshape(T, H, W, C)

    N = T - SEQ_LENGTH
    X_seq = np.stack([X_norm[i:i+SEQ_LENGTH] for i in range(N)], axis=0)
    y_seq = np.stack([y_arr[i+SEQ_LENGTH]   for i in range(N)], axis=0)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X_seq, y_seq, test_size=0.15, shuffle=False
    )
    val_frac = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac, shuffle=False
    )

    def to_tensor(X, y):
        X_t = torch.from_numpy(X).float().permute(0,1,4,2,3)
        y_t = torch.from_numpy(y).float().unsqueeze(1)
        return X_t, y_t

    X_tr, y_tr = to_tensor(X_train, y_train)
    X_va, y_va = to_tensor(X_val,   y_val)
    X_te, y_te = to_tensor(X_test,  y_test)

    # oversample fire sequences
    train_ds = TensorDataset(X_tr, y_tr)
    if OVERSAMPLE_FIRE and USE_TWO_HEAD:
        fire_mask = (y_tr > 0).view(len(y_tr), -1).any(dim=1)
        n_fire    = fire_mask.sum().item()
        n_non     = len(fire_mask) - n_fire
        weights   = torch.where(
            fire_mask,
            torch.tensor(n_non/(n_fire+1e-6), dtype=torch.float),
            torch.tensor(1.0)
        )
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    val_loader  = DataLoader(TensorDataset(X_va, y_va),
                             batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_te, y_te),
                             batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, C

def build_model(input_dim):
    model = ConvLSTM(
        input_dim=input_dim,
        hidden_dim=[HIDDEN_DIM]*NUM_LAYERS,
        kernel_size=KERNEL_SIZE,
        num_layers=NUM_LAYERS,
        batch_first=True,
        bias=True,
        return_all_layers=False
    ).to(DEVICE)

    # restore original head naming
    model.final_conv = nn.Conv2d(HIDDEN_DIM, 1, kernel_size=3, padding=1).to(DEVICE)
    model.reg_head   = model.final_conv

    if USE_TWO_HEAD:
        model.clf_head = nn.Conv2d(HIDDEN_DIM, 1, kernel_size=3, padding=1).to(DEVICE)

    return model

def train():
    train_loader, val_loader, _, C = prepare_data()

    # compute and clamp pos_weight for BCE
    fire_count = total = 0
    for _, yb in train_loader:
        mask = (yb > 0).float()
        fire_count += mask.sum().item()
        total += mask.numel()
    raw_pw = (total - fire_count) / (fire_count + 1e-6)
    pos_weight = min(raw_pw, MAX_POS_WEIGHT)
    print(f"Using pos_weight = {pos_weight:.2f} (raw was {raw_pw:.2f})")

    bce_loss  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=DEVICE))
    mse_elem  = nn.MSELoss(reduction="none")
    model     = build_model(C)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for ep in range(1, EPOCHS+1):
        model.train()
        run_loss = 0.0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()

            outs, _ = model(Xb)
            h       = outs[0][:, -1]
            logits  = model.clf_head(h)
            frp_hat = model.reg_head(h)
            mask_f  = (yb > 0).float()

            loss_clf = bce_loss(logits, mask_f)
            loss_reg = (mse_elem(frp_hat, yb) * mask_f).sum() / (mask_f.sum()+1e-6)
            loss     = loss_clf + REG_LOSS_WEIGHT * loss_reg

            loss.backward()
            optimizer.step()
            run_loss += loss.item() * Xb.size(0)

        train_loss = run_loss / len(train_loader.dataset)

        model.eval()
        val_run = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                outs, _ = model(Xb)
                h       = outs[0][:, -1]
                logits  = model.clf_head(h)
                frp_hat = model.reg_head(h)
                mask_f  = (yb > 0).float()

                loss_clf = bce_loss(logits, mask_f)
                loss_reg = (mse_elem(frp_hat, yb) * mask_f).sum() / (mask_f.sum()+1e-6)
                val_run += (loss_clf + REG_LOSS_WEIGHT*loss_reg).item() * Xb.size(0)

        val_loss = val_run / len(val_loader.dataset)
        print(f"Epoch {ep}/{EPOCHS} — Train: {train_loss:.4f}  Val: {val_loss:.4f}")

    torch.save(model.state_dict(), "convlstm_fire.pth")
    print("Saved model → convlstm_fire.pth")

if __name__ == "__main__":
    train()
