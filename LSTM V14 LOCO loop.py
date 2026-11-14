# =========================================================================
# Title: LSTM autoregressive forecasting loop code
# Version: 14.1
# Date: 02-06-2026
# Last rev: 12-11-2026
# Added in last rev: SHAP analysis
# Organisation: University of Twente, Electrical Engineering, Power Electronics
# Author: Maarten Mulder
# Description: Code uses excel input feature files to predict the SOH fade 
#              of a lithium test cell. Which cell is used to train the model
#              and which test can be set. A lookback window can also be set. 
#              The model will use this window for autoregressive forecasting.
# =========================================================================


import os
import sys
import traceback
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# Folder path where the script is saved
folder_path = r"c:\users\mjjmu\onedrive\bureaublad\bestanden\utwente\master\master thesis\software"
sys.path.append(folder_path)

# Import the worker function that contains the PyTorch LSTM and the autoregressive loop.
from lstm_worker import run_single_simulation

# =========================================================================
# 1. USER INPUT PARAMETERS
# =========================================================================
LOOKBACK_PERCENTAGE = 0.10   # Lookback winow percentage e.g: 0.10 = 10% lookback window.
TARGET_EOL_SOH = 85.0        # The SOH treshhold percentage to where the programming needs to predict e.g.: 70.0 = 70%

# Directories where the input data is saved and where the output data should be stored. 
data_dir = r"C:\Users\mjjmu\OneDrive\Bureaublad\LSTM Testmap"
output_dir = r"C:\Users\mjjmu\OneDrive\Bureaublad\LSTM Testmap\Results"

# Cells to be used for training the LSTM model
TRAIN_CELLS = ["Cell_13", "Cell_14", "Cell_16", "Cell_17", "Cell_18"]
# Cell to be tested, script automatically does LOCO on these cells.
TEST_CELLS  = ["Cell_15"]

TOTAL_RUNS = 1               # Amount of runs per cell to get a average. (due to random starting weights, results can differ)
MAX_WORKERS = 6              # Amount of CPU cores the program will spread calculations on.
EPOCHS = 80                  # Number of training loops for the neural network.
SHOW_PLOTS = False           # True = Automatically pop up matplotlib windows during the run.


target_col = "SOH_pct" # The variable the model is trying to predict.
EXCLUDE_FEATURES = []   # optional: columns to leave out, e.g. ["P_mean", "Z_phz_Deg"]
feature_cols = []       # filled automatically in the main block

os.makedirs(output_dir, exist_ok=True) # Create the results folder if it doesn't already exist.


# =========================================================================
# 2. HELPERS
# =========================================================================
# Loads the preprocessed feature dataset for a specific battery cell.
def extract_battery_dataframe(folder, file_name):
    full_path = os.path.join(folder, f"{file_name}_LSTM_Features.xlsx")
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Could not locate target Excel data sheet: {full_path}")
    print(f" -> Loading Excel sheet: {file_name}_LSTM_Features.xlsx")
    df = pd.read_excel(full_path) # Read the Excel file into a Pandas DataFrame.
    df.columns = df.columns.str.strip() # Strip any accidental whitespace from column names.
    df.fillna(0.0, inplace=True) # Replace any NaN's with 0.0 to prevent mathematical errors in the neural network.
    return df

# Decides which columns are used as model inputs. SOH is always first.
def select_feature_columns(dfs):
    first = next(iter(dfs.values()))
    common = set.intersection(*[set(d.columns) for d in dfs.values()])   # columns present in every cell
    if target_col not in common:
        sys.exit(f"Column '{target_col}' is missing in at least one Excel file.")
    if "EFC" not in common:
        sys.exit("Column 'EFC' is missing in at least one Excel file (needed for the plots).")
    cols = [c for c in first.columns
            if c in common and c != target_col
            and pd.api.types.is_numeric_dtype(first[c])
            and c not in EXCLUDE_FEATURES]
    skipped = [c for c in first.columns if c not in cols and c != target_col]
    if skipped:
        print(f" -> Not used as input: {skipped}")
    return [target_col] + cols

# Cuts off all data after the battery drops below the specified End-of-Life SOH.
def truncate_at_eol(df, threshold):
    below = df[df['SOH_pct'] < threshold] # Find all rows where SOH is lower than the target threshold.
    if not below.empty:
        return df.iloc[:below.index[0]].copy().reset_index(drop=True) # If it dropped below treshhold. End dataframe to that exact cycle.
    return df.copy() # If it does not reach treshold then return complete datafram.


# Converts a 2D spreadsheet into 3D sequences (Samples, TimeSteps, Features) for the LSTM.
def create_sequences_from_df(df, cols, tgt, feature_scaler, target_scaler, lookback):
    scaled_feats = feature_scaler.transform(df[cols]) # Scale input features between 0 and 1.
    scaled_targs = target_scaler.transform(df[[tgt]]).squeeze() # Scale ouputs to be between 0 and 1.
    X, y = [], []
    for i in range(len(scaled_feats) - lookback): # Slide a window of size 'lookback' over the data, one row at a time.
        X.append(scaled_feats[i: i + lookback]) # Input X is a block of past cycles. e.g: cycle 0 to 80. 
        y.append(scaled_targs[i + lookback]) # Output Y is the SOH of the very next cycle. e.g: 81 in this case.
    return np.array(X), np.array(y) # Convert lists to numpy arrays. 


# =========================================================================
# 3. ONE FOLD (= one test cell)
# =========================================================================
# Handles the entire pipeline (scaling, training, predicting, scoring) for a single test cell.
def run_fold(test_name, all_dfs, executor):
    train_names = [c for c in TRAIN_CELLS if c != test_name] # Exclude the current test cell from the training list. 
    print("\n" + "#" * 65)
    print(f" FOLD: test = {test_name} | train = {', '.join(train_names)}")
    print("#" * 65)

    train_dfs = [all_dfs[c] for c in train_names] # Load the pre-loaded dataframes for testing and training. 
    df_test_full = all_dfs[test_name]
    df_test_trunc = truncate_at_eol(df_test_full, TARGET_EOL_SOH) # Truncate the test cell at the target SOH so we only predict up to that point

    # Combine all training cells into one massive dataframe to fit the scaler.
    train_combined = pd.concat(train_dfs, axis=0)
    scaler_features = MinMaxScaler().fit(train_combined[feature_cols])
    scaler_target = MinMaxScaler().fit(train_combined[[target_col]])

    # Calculate the number of cycles the LSTM model will look at based on the test cell's length .
    lookback = int(len(df_test_trunc) * LOOKBACK_PERCENTAGE)
    print(f" -> Test length: {len(df_test_trunc)} cycles | lookback: {lookback} | "
          f"prediction window: {len(df_test_trunc) - lookback}")

    # Process all training files into a large 3D LSTM sequence. 
    X_list, y_list = [], []
    for name, df in zip(train_names, train_dfs):
        if len(df) <= lookback: # If a training battery died faster then the lookback window, skip it to avoid errors.
            print(f" -> WARNING: Skipping {name}. Length ({len(df)}) is shorter than lookback ({lookback}).")
            continue
        
        # Create sliding window for this training cell.
        Xp, yp = create_sequences_from_df(df, feature_cols, target_col,
                                          scaler_features, scaler_target, lookback)
        X_list.append(Xp)
        y_list.append(yp)
        
    # Stack all training sequences from all cells together.
    X_train_arr = np.concatenate(X_list, axis=0)
    y_train_arr = np.concatenate(y_list, axis=0)

    #Scale the test features. These will be fed to the worker for lookback and EFC updates. 
    scaled_test = scaler_features.transform(df_test_trunc[feature_cols])

    # Divide the training runs on multiple CPU cores. 
    futures = [
        executor.submit(run_single_simulation, i, X_train_arr, y_train_arr, scaled_test,
                        df_test_trunc, feature_cols, target_col, lookback, EPOCHS, test_name, output_dir)
        for i in range(TOTAL_RUNS)
    ]
    all_runs_preds = [f.result() for f in futures]# Wait for all runs to finish and collect their final SOH predictions.

    # Extract the true SOH values from the test cell after the lookback window.
    actual = df_test_trunc[target_col].values[lookback:].reshape(-1, 1)
    
    # Extract the true SOH values from the test cell, strictly AFTER the lookback window.
    real_preds_runs = [scaler_target.inverse_transform(np.array(p).reshape(-1, 1)) for p in all_runs_preds]
    
    # Calculate Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for every run.
    mae = [np.mean(np.abs(p - actual)) for p in real_preds_runs]
    rmse = [np.sqrt(np.mean((p - actual) ** 2)) for p in real_preds_runs]
    
    avg_pred = np.mean(real_preds_runs, axis=0).ravel() # Calculate average of all runs.
    std_pred = np.std(real_preds_runs, axis=0).ravel() # Calculate the standard devatiation.
    end_err = float(avg_pred[-1] - actual[-1, 0]) # Calculat how far the of the final prediction is from the true EOL. 

    print(f" -> MAE : {np.mean(mae):.3f} % SOH (± {np.std(mae):.3f})")
    print(f" -> RMSE: {np.mean(rmse):.3f} % SOH (± {np.std(rmse):.3f})")
    print(f" -> Error at end of window: {end_err:+.3f} % SOH")

    # Save a CSV containing the exact true SOH and predicted SOH.
    efc_pred = df_test_trunc["EFC"].values[lookback:]
    pd.DataFrame({
        "EFC": efc_pred, "True_SOH": actual.ravel(),
        "Pred_SOH_mean": avg_pred, "Pred_SOH_std_over_runs": std_pred,
    }).to_csv(os.path.join(output_dir, f"{test_name}_predictions.csv"), index=False)

    # Get the X and Y axes for the global trajectory. 
    full_efc = df_test_full["EFC"].values[lookback:]
    full_soh = df_test_full[target_col].values[lookback:]
    plt.figure(figsize=(10, 6))
    
    # Plot the full ground truth degradation curve.
    plt.plot(full_efc, full_soh, label="True SOH (Full Lifespan)", color="blue", linewidth=2)
    
    # Plot the LSTM's autoregressive forecast line.
    plt.plot(efc_pred, avg_pred, label=f"Predicted SOH (Up to {TARGET_EOL_SOH}%)",
             color="red", linestyle="--", linewidth=2)
    
    # Shade the area between the different runs in the plot. 
    plt.fill_between(efc_pred, avg_pred - std_pred, avg_pred + std_pred, color="red", alpha=0.15)
    
    #Draws a dotted line to show where the dataset was forced to truncate. 
    plt.axhline(y=TARGET_EOL_SOH, color="gray", linestyle=":", label=f"Prediction Cutoff ({TARGET_EOL_SOH}% SOH)")
    plt.title(f"Battery SOH Forecasting vs True Degradation — {test_name}", fontsize=14, fontweight="bold")
    plt.xlabel("Equivalent Full Cycles (EFC)", fontsize=12)
    plt.ylabel("State of Health (SOH %)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{test_name}_SOH_Forecast_Plot_EOL_{TARGET_EOL_SOH}.png")
    plt.savefig(save_path, dpi=300)
    print(f" -> Plot saved to: {save_path}")
    
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    # Return summary statistics.
    return {
        "summary": {
            "Test_Cell": test_name,
            "MAE": np.mean(mae), "MAE_std": np.std(mae),
            "RMSE": np.mean(rmse), "RMSE_std": np.std(rmse),
            "End_Error": end_err,
            "Forecast_Cycles": len(avg_pred),
        },
        "plot": (full_efc, full_soh, efc_pred, avg_pred, std_pred),
    }


# =========================================================================
# 4. MAIN: LOAD ONCE, RUN ALL FOLDS, SUMMARISE
# =========================================================================
if __name__ == '__main__':
    # Ensure every test cell actually has at least one training cell assigned to it.
    for name in TEST_CELLS:
        remaining = [c for c in TRAIN_CELLS if c != name]
        if not remaining:
            sys.exit(f"Test cell {name} has no training cells left. Add more cells to TRAIN_CELLS.")
            
    # Compile a list of every cell needed for this script to run.
    needed = list(dict.fromkeys(TRAIN_CELLS + TEST_CELLS))
    print(f"Train cells: {TRAIN_CELLS}\nTest cells : {TEST_CELLS}\nLoading cells...")
    
    # Extract the excel file once and store in memory to save time. 
    all_dfs = {c: extract_battery_dataframe(data_dir, c) for c in needed}
    
    feature_cols = select_feature_columns(all_dfs)
    print(f"\n -> {len(feature_cols)} input features: {feature_cols}")

    results = {}
    # Used for multi core processing.
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Loop over every test cell and run its evaluation.
        for test_name in TEST_CELLS:
            try:
                results[test_name] = run_fold(test_name, all_dfs, executor)
            except Exception:
                # If a cell crashes, cathc the error, print it, and recover to go to the next cell.  
                print(f"\n!! Fold {test_name} FAILED, continuing with the next one:")
                traceback.print_exc()

    if not results:
        sys.exit("No folds completed.")

    # Create a pandas dataframe from all the returned summary dictionaries.
    summary = pd.DataFrame([r["summary"] for r in results.values()])
    
    # Calculate the average MAE, RMSE, etc for all test cells. 
    mean_row = {"Test_Cell": "MEAN"}
    for col in ["MAE", "MAE_std", "RMSE", "RMSE_std", "End_Error", "Forecast_Cycles"]:
        mean_row[col] = summary[col].mean()
        
    # Append the mean row to the bottem of the dataframe.
    summary_out = pd.concat([summary, pd.DataFrame([mean_row])], ignore_index=True)
    
    # Save the global CSV to the disk.
    summary_path = os.path.join(output_dir, "LOCO_summary.csv")
    summary_out.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print(f" SUMMARY ({TOTAL_RUNS} runs per test cell)")
    print("=" * 65)
    print(summary_out.round(3).to_string(index=False)) # Print the table to the console.
    print(f"\n -> Summary saved to: {summary_path}")

    # Calculate how many rows and columns are needed in the Matplotlib grid.
    n = len(results)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    
    # Create a massive master figure.
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.2 * nrows), squeeze=False)
    
    # Loop over every tested cell and plot its individual results on a subplot.
    for ax, (name, r) in zip(axes.ravel(), results.items()):
        full_efc, full_soh, efc_pred, avg_pred, std_pred = r["plot"]
        ax.plot(full_efc, full_soh, color="blue", linewidth=2, label="True")
        ax.plot(efc_pred, avg_pred, color="red", linestyle="--", linewidth=2, label="Predicted")
        ax.fill_between(efc_pred, avg_pred - std_pred, avg_pred + std_pred, color="red", alpha=0.15)
        ax.set_title(f"{name} (MAE {r['summary']['MAE']:.2f})")
        ax.set_xlabel("EFC")
        ax.set_ylabel("SOH (%)")
        ax.grid(True, linestyle="--", alpha=0.5)
        
    # Turn off any leftover empty subplots if the total count isn't perfectly divisible by 3.
    for ax in axes.ravel()[n:]:
        ax.axis("off")
        
    # Put a legend on the very first subplot as a reference.
    axes.ravel()[0].legend()
    fig.tight_layout()
    
    # Save the giant overview image containing every battery chart side-by-side.
    overview_path = os.path.join(output_dir, "LOCO_overview.png")
    fig.savefig(overview_path, dpi=200)
    print(f" -> Overview figure saved to: {overview_path}")
    if SHOW_PLOTS:
        plt.show()