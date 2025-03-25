import numpy as np
import xarray as xr
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# Load NetCDF data
ds = xr.open_dataset('your_data.nc')

# Input features (shape: time × lat × lon)
input_vars = ['d2m', 'u10', 'v10', 'tp', 'lai_hv', 'lai_lv']
X = np.stack([ds[var].values for var in input_vars], axis=-1)  # (744, 8, 16, 6)
y = ds['frp'].values  # (744, 8, 16)

# Normalize features per variable
scaler = StandardScaler()
X_norm = scaler.fit_transform(X.reshape(-1, 6)).reshape(744, 8, 16, 6)

# Reshape into sequences (e.g., 24-hour windows)
seq_length = 24  # 1-day sequences
n_samples = 744 - seq_length
X_seq = np.array([X_norm[i:i+seq_length] for i in range(n_samples)])  # (720, 24, 8, 16, 6)
y_seq = np.array([y[i+seq_length] for i in range(n_samples)])         # (720, 8, 16)

input_dim = 6      # Channels (d2m, u10, v10, tp, lai_hv, lai_lv)
hidden_dim = 64    # Number of hidden channels
kernel_size = (3, 3) 
num_layers = 2     # Stacked ConvLSTM layers
batch_first = True # Input shape starts with batch_size

model = ConvLSTM(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    kernel_size=kernel_size,
    num_layers=num_layers,
    batch_first=batch_first,
    bias=True,
    return_all_layers=False
)

# Add a final convolutional layer to map hidden_dim to output (8, 16)
model.final_conv = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)  # Output 1 channel (FRP)