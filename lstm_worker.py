# =========================================================================
# Title: LSTM worker
# Version: 1.1
# Date: 02-10-2026
# Last rev: 12-11-2026
# Added in last rev: SHAP analysis
# Organisation: University of Twente, Electrical Engineering, Power Electronics
# Author: Maarten Mulder
# Description: Worker code for the LSTM multi CPU core code.
#              Is a different file so the main code can fire it up multiple times.
# =========================================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import shap
import matplotlib
matplotlib.use("Agg")            # Worker processes have no screen: always draw to file.
import matplotlib.pyplot as plt
import os

# =========================================================================
# SETTINGS
# =========================================================================
RELATIVE_SOH           = True    # show the SOH channel as. (SOH - latest SOH in window)
INPUT_NOISE            = 0.002   # std of Gaussian noise on the SOH channel during training. (scaled units)
ENFORCE_NON_INCREASING = True   # SOH is never allowed to rise during the rollout.
SOH_IDX                = 0       # SOH_pct must be feature column 0. (your main script already does this)
DO_SHAP_ANALYSIS       = True    # Generate SHAP explanation plot on Run 1


# The PyTorch Neural Network Architecture.
class BatteryLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, output_dim=1, dropout=0.2):
        super(BatteryLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # The core LSTM memory cells. batch_first=True means data comes in as samples, timeSteps, features.
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # Dropout randomly turns off neurons during training to prevent memorization. (overfitting)
        self.dropout = nn.Dropout(dropout)
        
        # The final mathematical layer that crushes the high-dimensional hidden state down to a single number. (the prediction)
        self.fc = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        
        # Initialize the short-term memory (h0) and long-term cell state (c0) with zeros.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Pass the input data and the blank memories through the LSTM layer.
        out, _ = self.lstm(x, (h0, c0))
        
        # Only care about the network's state at the last time step of the window to make the future prediction.
        out = self.dropout(out[:, -1, :])
        
        # Pass that final state through the linear layer to get our exact SOH delta prediction.
        return self.fc(out)


# Optionally express the SOH channel relative to the last SOH value in each window.
def _prep(x):
    if not RELATIVE_SOH:
        return x
    
    # Clone the data so the code doesn't accidentally modify the original dataset in memory.
    x = x.clone()
    
    # Subtract the last SOH value in the window from EVERY SOH value in the window.
    # This forces the window to always end exactly at 0.0, teaching the model to focus on the 'slope' of the curve.
    x[:, :, SOH_IDX] = x[:, :, SOH_IDX] - x[:, -1:, SOH_IDX]
    return x


# Trains the model and executes the autoregressive rollout for a single CPU thread.
def run_single_simulation(run_id, X_train_arr, y_train_arr, scaled_test_features_all,
                          df_test_raw, feature_cols, target_col, lookback, epochs, test_name="UnknownCell", output_dir=""):
    print(f" -> Start run {run_id + 1}...")
    # Seed the random number generators so that Run 1 is always exactly the same if you run the script twice, 
    # but Run 2 will get different initial weights to create ensemble variance.
    torch.manual_seed(run_id)
    np.random.seed(run_id)


    # ---------------------------------------------------------------------
    # Targets: change in (scaled) SOH per cycle, standardised.
    # ---------------------------------------------------------------------
    # Convert the Numpy arrays from the main script into PyTorch Tensors.
    X_train = torch.tensor(X_train_arr, dtype=torch.float32)
    y_abs = torch.tensor(y_train_arr, dtype=torch.float32)
    
    # RESIDUAL LEARNING: Instead of predicting the absolute future SOH, we calculate the DELTA. (the drop).
    # Delta = True Future SOH - Current SOH.
    delta = y_abs - X_train[:, -1, SOH_IDX]     

    # Standardize the deltas (z-score) so they have a mean of 0 and a standard deviation of 1.
    # Neural networks learn exponentially faster when targets are centered around zero.    
    d_mean, d_std = delta.mean(), delta.std() + 1e-9
    z = ((delta - d_mean) / d_std).unsqueeze(-1)

    # Initialize the actual neural network, the loss function (Mean Squared Error), and the Adam optimizer. (learning rate)
    model = BatteryLSTM(input_dim=len(feature_cols), hidden_dim=64, num_layers=2, dropout=0.2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Package the training data into a PyTorch DataLoader to feed it to the model in batches of 16 sequences at a time.
    train_loader = DataLoader(TensorDataset(_prep(X_train), z), batch_size=16, shuffle=True)

    # Put the model into training mode. (enables Dropout and Gradient Tracking)
    model.train()
    for epoch in range(1, epochs + 1):
        for batch_X, batch_z in train_loader:
            if INPUT_NOISE > 0:
                # Add tiny random Gaussian noise to the SOH input to build robustness.
                batch_X = batch_X.clone()
                batch_X[:, :, SOH_IDX] += INPUT_NOISE * torch.randn_like(batch_X[:, :, SOH_IDX])
                
            # Wipe the gradients from the previous batch.
            optimizer.zero_grad()
            # Feed the window through the network and calculate how wrong the prediction was.
            loss = criterion(model(batch_X), batch_z)
            # Calculate the mathematical derivative (gradient) of the error for every weight in the network.
            loss.backward()
            # Nudge the weights in the correct direction to make the error smaller next time.
            optimizer.step()


# ---------------------------------------------------------------------
    # SHAP ANALYSIS (Only on Run 1)
    # ---------------------------------------------------------------------
    if DO_SHAP_ANALYSIS and run_id == 0:
        print(f" -> [Run 1] Generating SHAP Analysis for {test_name}. This may take a minute...")
        model.eval()
        
        # 1. Create a wrapper to combine _prep and the model so SHAP understands the relative math
        class SHAPWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return self.m(_prep(x))
                
        shap_model = SHAPWrapper(model)
        
        # 2. Select background data (baseline) from training set
        bg_idx = np.random.choice(X_train.shape[0], min(100, X_train.shape[0]), replace=False)
        background = X_train[bg_idx]
        
        # 3. Reconstruct true sequences from the Unseen Test Cell to see why it fails
        X_test_list = []
        for i in range(len(scaled_test_features_all) - lookback):
            X_test_list.append(scaled_test_features_all[i : i + lookback])
        X_test_tensor = torch.tensor(np.array(X_test_list), dtype=torch.float32)
        
        test_idx = np.random.choice(X_test_tensor.shape[0], min(100, X_test_tensor.shape[0]), replace=False)
        test_samples = X_test_tensor[test_idx]
        
        # 4. Run the DeepExplainer
        explainer = shap.GradientExplainer(shap_model, background)
        shap_values = explainer.shap_values(test_samples)
        # Make the output shape version-proof: -> (samples, timesteps, features).
        # Older SHAP returns an array or a list with one array; newer SHAP adds a trailing output axis.
        sv = shap_values[0] if isinstance(shap_values, list) else shap_values
        sv = np.asarray(sv)
        if sv.ndim == 4:
            sv = sv[..., 0]

        # The model reads the WHOLE window, so add up each feature's contribution over all timesteps.
        sv_feat = sv.sum(axis=1)                          # (samples, features)
        vals_feat = test_samples.numpy().mean(axis=1)     # window-mean value per feature, used for colouring

        # Beeswarm plot
        plt.figure(figsize=(10, max(4, 0.4 * len(feature_cols) + 2)))
        shap.summary_plot(sv_feat, vals_feat, feature_names=feature_cols,
                  max_display=len(feature_cols), show=False)
        plt.title(f"SHAP Feature Effects on next-cycle SOH drop (Unseen Data: {test_name})", fontsize=13)
        plt.tight_layout()
        shap_path = os.path.join(output_dir, f"{test_name}_SHAP_Summary.png")
        plt.savefig(shap_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Bar plot of mean |SHAP| + csv with the numbers
        mean_abs = np.abs(sv_feat).mean(axis=0)
        order = np.argsort(mean_abs)
        plt.figure(figsize=(10, max(4, 0.4 * len(feature_cols) + 2)))
        plt.barh([feature_cols[i] for i in order], mean_abs[order], color="tab:blue")
        plt.xlabel("Mean |SHAP value| (standardised delta-SOH)")
        plt.title(f"SHAP Feature Importance (Unseen Data: {test_name})", fontsize=13)
        plt.grid(True, axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{test_name}_SHAP_Importance.png"), dpi=300, bbox_inches='tight')
        plt.close()

        pd.DataFrame({"Feature": feature_cols, "Mean_abs_SHAP": mean_abs}) \
        .sort_values("Mean_abs_SHAP", ascending=False) \
        .to_csv(os.path.join(output_dir, f"{test_name}_SHAP_Importance.csv"), index=False)
        print(f" -> SHAP plots + csv saved to: {output_dir}")
        
       
        print(f" -> SHAP Plot saved to: {shap_path}")


    # ---------------------------------------------------------------------
    # Rollout: start from the TRUE history, add one predicted delta per step.
    # ---------------------------------------------------------------------
    # Put the model into evaluation mode. (turns off Dropout so predictions are deterministic)
    model.eval()
    
    # Grab the very first true lookback window from the test cell. (e.g., Cycles 0 to 80)
    window = torch.tensor(scaled_test_features_all[:lookback], dtype=torch.float32).unsqueeze(0)
    
    # Calculate exactly how many cycles into the future we need to predict.
    total_forecast_steps = len(df_test_raw) - lookback
    preds_scaled = []

    # Turn off gradient tracking to save massive amounts of RAM and CPU power during prediction.
    with torch.no_grad():
        for step in range(total_forecast_steps):
            
            # Predict the standardized SOH drop (z-score) for the next cycle.
            z_hat = model(_prep(window)).item()
            
            # Un-standardize the prediction back into the scaled normal range.
            d = z_hat * d_std.item() + d_mean.item()
            
            # Apply physics constraint: SOH drop cannot be positive. (battery can't heal)
            if ENFORCE_NON_INCREASING:
                d = min(d, 0.0)
                
            # Calculate the new absolute SOH by subtracting the drop from the current SOH.
            soh_next = window[0, -1, SOH_IDX].item() + d
            preds_scaled.append(soh_next)

            # Fetch the known future covariates (like EFC) for the next cycle.
            nxt = torch.tensor(scaled_test_features_all[lookback + step], dtype=torch.float32)
            # Overwrite the TRUE unknown future SOH with our model's PREDICTED SOH.
            nxt[SOH_IDX] = soh_next
            
            # Slide the window forward: Drop cycle 0, and append the newly predicted cycle to the end.
            window = torch.cat([window[:, 1:, :], nxt.view(1, 1, -1)], dim=1)

    print(f" -> Run {run_id + 1} voltooid!")
    return preds_scaled