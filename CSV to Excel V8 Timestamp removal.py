# =========================================================================
# Title: CSV Arbin data to Excel input features converter
# Version: 8.0
# Date: 25-06-2026
# Last rev: 06-10-2026
# Added in last rev: Hybrid timestamp removal
# Organisation: University of Twente, Electrical Engineering, Power Electronics
# Author: Maarten Mulder
# Description: Converts the workbook (WB) files created by the Arbin tester
#              and the ACIM files created by the Gamry into an Excel input
#              features file. Set the file directory of the folder these 
#              files are in and set the metadata per cell and it is set up.
#              You can set manual timestamp removal, but the program does
#              also automatically do this.
# =========================================================================

import os
import glob
import re
import sys
import numpy as np
import pandas as pd

# =========================================================================
# 1. INPUT PARAMETERS
# =========================================================================
folder_path = r"C:\Users\mjjmu\OneDrive\Bureaublad\LSTM Testmap" # File directory where you WB and DCIM files are
output_cell_name = "Cell_18"  # Specify the name of the files. e.g: "Cell_14_NMC_Ageing_C10_Ch_2_Wb_1.CSV"

# Per cell metadata
CELL_METADATA = {
    "C_rate": 1.0,           
    "Temperature": 25.0,     
    "DoD": 80.0,             
    "Lower_SOC": 20.0,
    "Upper_SOC": 100.0,
    "Nominal_Capacity": 4.5  
}

# Comment out any feature to remove from the outputted excel file
FEATURES_TO_EXPORT = [
    "EFC", 
    "SOH_pct", 
    #"DCIR_Ohms", 
    "Re_Ohms", 
    "Rct_Ohms", 
    "ACR_Ohms",
    "Z_mod_Ohms", 
    "Z_phz_Deg", 
    "Z_real_Ohms", 
    "Z_img_Ohms",
    "Discharge_Capacity_Ah", 
    "Charge_Capacity_Ah", 
    #"Coulombic_Efficiency_pct",
    #"OCV_V", 
    #"V_max", 
    #"V_mean", 
    #"I_charge_mean", 
    #"I_discharge_mean", 
    #"P_max", 
    "P_mean", 
    "Charge_Energy_Wh", 
    "Discharge_Energy_Wh",
    #"dQ_dV_max_Ah_V", 
    "dV_dt_max_V_s",
    "C_rate", 
    "Temperature", 
    "DoD", 
    "Lower_SOC", 
    "Upper_SOC"
]

output_excel_path = os.path.join(folder_path, f"{output_cell_name}_LSTM_Features.xlsx") # Output folder en file name

# =========================================================================
# 2. LOAD DATA & CALCULATE TIME DELTAS
# =========================================================================
print(f"Scanning directory for Workbook logs matching cell: {output_cell_name}...")
wb_pattern = os.path.join(folder_path, f"*{output_cell_name}*_Wb_*.CSV")
wb_files = glob.glob(wb_pattern) # Scan and grab every file in the folder including WB 
wb_files.sort(key=lambda f: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', f)]) # Sort the WB files in order

if not wb_files:
    print(f"CRITICAL ERROR: No workbook files found with given name")
    sys.exit()

wb_list = []
for file in wb_files:
    df_chunk = pd.read_csv(file, sep=None, engine='python') # Load each WB chunk into a individual pandas dataframe
    df_chunk.columns = df_chunk.columns.str.strip()
    
    if 'Test Time (s)' in df_chunk.columns:
        df_chunk['Test Time (s)'] = pd.to_numeric(df_chunk['Test Time (s)'], errors='coerce').fillna(0.0)
        df_chunk['dt_sec'] = df_chunk['Test Time (s)'].diff().fillna(0.0) # Calculate timestemps between every single row
        df_chunk.loc[df_chunk['dt_sec'] > 20.0, 'dt_sec'] = 0.0 # Safety net to never have a timestamp of more then 20 seconds. 
    else:
        df_chunk['dt_sec'] = 0.0
        
    if 'Date Time' in df_chunk.columns:
        df_chunk['Date Time'] = pd.to_datetime(df_chunk['Date Time'], errors='coerce')
        
    wb_list.append(df_chunk)

df_wb = pd.concat(wb_list, axis=0, ignore_index=True) # Make one big dataframe from all the workbooks 

for col in ['Discharge Capacity (Ah)', 'Charge Capacity (Ah)', 'Voltage (V)', 'Current (A)']:
    if col in df_wb.columns:
        df_wb[col] = pd.to_numeric(df_wb[col], errors='coerce').fillna(0.0) # Remove any impossible inputs to NaN

acim_pattern = os.path.join(folder_path, f"*{output_cell_name}*_ACIM.CSV")
acim_files = glob.glob(acim_pattern)
acim_files.sort(key=lambda f: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', f)]) 

if not acim_files:
    print(f"CRITICAL ERROR: Could not locate the ACIM file")
    sys.exit()

acim_list = []
for file in acim_files:
    temp_df = pd.read_csv(file) # Load the ACIM file into a dataframe
    temp_df.columns = temp_df.columns.str.strip()
    acim_list.append(temp_df)

df_acim = pd.concat(acim_list, axis=0, ignore_index=True)

if 'Cycle_ID' in df_acim.columns:
    cycle_drops = df_acim['Cycle_ID'].diff() < 0
    while cycle_drops.any():
        drop_idx = cycle_drops.idxmax()
        max_cycle_reached = df_acim.loc[drop_idx - 1, 'Cycle_ID']
        current_reset_val = df_acim.loc[drop_idx, 'Cycle_ID']
        shift_amount = max_cycle_reached - current_reset_val + 1
        df_acim.loc[drop_idx:, 'Cycle_ID'] += shift_amount
        cycle_drops = df_acim['Cycle_ID'].diff() < 0

# =========================================================================
# 3. FEATURE EXTRACTION
# =========================================================================
print("\nExtracting physical sensors, ACIM impedance, and Signal Derivatives...")

structured_rows = []
grouped_wb = df_wb.groupby('Cycle Index') # Divide huge dataframe in blocks based on cycle numbers

initial_capacity = None
cumulative_discharge_ah = 0.0

# Go trough every cycle
for cycle_idx in sorted(grouped_wb.groups.keys()): 
    group = grouped_wb.get_group(cycle_idx)
    
    # Set all features to NaN before filling in first features set
    current_row = {
        "Cycle_Index": cycle_idx, 
        "Cycle_End_Time": group['Date Time'].iloc[-1] if 'Date Time' in group.columns else pd.NaT,
        "EFC": 0.0, "SOH_pct": np.nan, "DCIR_Ohms": np.nan, "Re_Ohms": np.nan, "Rct_Ohms": np.nan,
        "Discharge_Capacity_Ah": np.nan, "Charge_Capacity_Ah": np.nan, "Coulombic_Efficiency_pct": np.nan,
        "V_max": np.nan, "V_mean": np.nan, "I_charge_mean": np.nan, "I_discharge_mean": np.nan,
        "P_max": np.nan, "P_mean": np.nan, "Charge_Energy_Wh": np.nan, "Discharge_Energy_Wh": np.nan,
        "OCV_V": np.nan, "ACR_Ohms": np.nan, "Z_mod_Ohms": np.nan, "Z_phz_Deg": np.nan,
        "Z_real_Ohms": np.nan, "Z_img_Ohms": np.nan, "dQ_dV_max_Ah_V": np.nan, "dV_dt_max_V_s": np.nan,
        "C_rate": float(CELL_METADATA["C_rate"]), "Temperature": float(CELL_METADATA["Temperature"]),
        "DoD": float(CELL_METADATA["DoD"]), "Lower_SOC": float(CELL_METADATA["Lower_SOC"]),
        "Upper_SOC": float(CELL_METADATA["Upper_SOC"])
    }
    
    # OCV extraction
    d_mask = group['Current (A)'] < -0.01 # Find where discharge load begins
    if d_mask.any():
        first_d_idx = d_mask.idxmax() 
        loc_in_group = group.index.get_loc(first_d_idx)
        if loc_in_group > 0:
            current_row["OCV_V"] = group.iloc[loc_in_group - 1]['Voltage (V)']
        else:
            current_row["OCV_V"] = group.iloc[loc_in_group]['Voltage (V)']

    # SOH & EFC Tracking
    d_cap = group['Discharge Capacity (Ah)'].max() if 'Discharge Capacity (Ah)' in group.columns else 0.0
    c_cap = group['Charge Capacity (Ah)'].max() if 'Charge Capacity (Ah)' in group.columns else 0.0
    cycle_cap = d_cap
    
    if cycle_cap > 1.0: 
        v_min = group['Voltage (V)'].min()
        cumulative_discharge_ah += cycle_cap  
        current_row["EFC"] = cumulative_discharge_ah / CELL_METADATA["Nominal_Capacity"]
        
        if v_min < 3.1: # Check wheter it was a full by comparing the minimum discharge voltage
            if initial_capacity is None:
                initial_capacity = cycle_cap
            current_row["SOH_pct"] = (cycle_cap / initial_capacity) * 100.0 # Update the SOH percentage
        else: # The cycle was not a full cycle so continue with normal calculations
            c_mask = group['Current (A)'] > 0.01
            true_d_cap = np.sum((np.abs(group.loc[d_mask, 'Current (A)']) * group.loc[d_mask, 'dt_sec']) / 3600.0)
            true_c_cap = np.sum((group.loc[c_mask, 'Current (A)'] * group.loc[c_mask, 'dt_sec']) / 3600.0)
            true_d_energy = np.sum((np.abs(group.loc[d_mask, 'Current (A)']) * group.loc[d_mask, 'Voltage (V)'] * group.loc[d_mask, 'dt_sec']) / 3600.0)
            true_c_energy = np.sum((group.loc[c_mask, 'Current (A)'] * group.loc[c_mask, 'Voltage (V)'] * group.loc[c_mask, 'dt_sec']) / 3600.0)
            
            current_row["Discharge_Capacity_Ah"] = true_d_cap
            current_row["Charge_Capacity_Ah"] = true_c_cap
            current_row["Discharge_Energy_Wh"] = true_d_energy
            current_row["Charge_Energy_Wh"] = true_c_energy
            
            if true_c_cap > 0.01:
                current_row["Coulombic_Efficiency_pct"] = (true_d_cap / true_c_cap) * 100.0
            
            current_row["V_max"] = group['Voltage (V)'].max()
            current_row["V_mean"] = group['Voltage (V)'].mean()
            current_row["I_charge_mean"] = group.loc[c_mask, 'Current (A)'].mean() if not group[c_mask].empty else 0.0
            current_row["I_discharge_mean"] = np.abs(group.loc[d_mask, 'Current (A)'].mean()) if not group[d_mask].empty else 0.0
            
            power_array = group['Voltage (V)'] * np.abs(group['Current (A)'])
            current_row["P_max"] = power_array.max()
            current_row["P_mean"] = power_array.mean()

            # dQ/dV and dV/dt Derivations
            dV = group['Voltage (V)'].diff().fillna(0.0)
            dt_sec = group['dt_sec']
            dQ_Ah = (np.abs(group['Current (A)']) * dt_sec) / 3600.0
            
            valid_dt = dt_sec > 0.5
            if valid_dt.any():
                dV_dt = np.abs(dV[valid_dt] / dt_sec[valid_dt])
                current_row["dV_dt_max_V_s"] = dV_dt.max()
                
            valid_dV = np.abs(dV) > 0.005 
            if valid_dV.any():
                dQ_dV_raw = dQ_Ah[valid_dV] / np.abs(dV[valid_dV])
                dQ_dV_smooth = dQ_dV_raw.rolling(window=5, min_periods=1).mean()
                current_row["dQ_dV_max_Ah_V"] = dQ_dV_smooth.max()

    # DCIR extraction
    if 'Internal Resistance' in group.columns and group['Internal Resistance'].notna().any():
        valid_res = group['Internal Resistance'][group['Internal Resistance'] > 0]
        if not valid_res.empty:
            current_row["DCIR_Ohms"] = float(valid_res.mean())

    # ACIM 1kHz Extraction
    cycle_eis = df_acim[df_acim['Cycle_ID'] == cycle_idx]
    if not cycle_eis.empty:
        capacitive_eis = cycle_eis[cycle_eis['Zimg'] <= 0] # Filter out the inductive high-frequency tail (where Zimg > 0)
        if capacitive_eis.empty: # Fallback just in case the whole sweep was shifted
            capacitive_eis = cycle_eis
            
        z_real_vals_clean = capacitive_eis['Zreal'].values
        valid_z_real = z_real_vals_clean[z_real_vals_clean > 0] # Secondary safety net: strictly forbid negative real resistance
        
        if len(valid_z_real) > 0:
            current_row["Re_Ohms"] = float(np.min(valid_z_real))
            current_row["Rct_Ohms"] = float(np.max(valid_z_real) - current_row["Re_Ohms"])
        else:
            current_row["Re_Ohms"] = np.nan
            current_row["Rct_Ohms"] = np.nan
        
        freq_col = [c for c in cycle_eis.columns if 'freq' in c.lower() or 'hz' in c.lower()]
        if freq_col:
            target_freq = 1000.0
            idx_1khz = (np.abs(cycle_eis[freq_col[0]] - target_freq)).argmin()
            row_1khz = cycle_eis.iloc[idx_1khz]
            
            if 'Zreal' in row_1khz: 
                current_row["Z_real_Ohms"] = row_1khz['Zreal']
                current_row["ACR_Ohms"] = row_1khz['Zreal']
            if 'Zimg' in row_1khz: current_row["Z_img_Ohms"] = row_1khz['Zimg']
            if 'Zmod' in row_1khz: current_row["Z_mod_Ohms"] = row_1khz['Zmod']
            if 'Zphz' in row_1khz: current_row["Z_phz_Deg"] = row_1khz['Zphz']
        
    structured_rows.append(current_row)

df_out = pd.DataFrame(structured_rows) # Convert all the lists to one 2D Pandas list

# =========================================================================
# 4. INITIAL INTERPOLATION
# =========================================================================
smooth_columns = [
    "SOH_pct", "EFC", "DCIR_Ohms", "Discharge_Capacity_Ah", "Charge_Capacity_Ah", 
    "Coulombic_Efficiency_pct", "V_max", "V_mean", "I_charge_mean", "I_discharge_mean", 
    "P_max", "P_mean", "Charge_Energy_Wh", "Discharge_Energy_Wh", "OCV_V",
    "dQ_dV_max_Ah_V", "dV_dt_max_V_s"
]

for col in smooth_columns: # Smooth out features that are not available every cycle by interpolating and filling blank spots at end and beginning
    if col in df_out.columns:
        df_out[col] = df_out[col].replace(0.0, np.nan).interpolate(method='linear').ffill().bfill()

if "EFC" in df_out.columns and df_out["EFC"].isna().iloc[0]:
    df_out["EFC"] = np.arange(1, len(df_out) + 1) * (CELL_METADATA["DoD"] / 100.0)

# =========================================================================
# 3b. REMOVE BAD RANGE + STITCH (manual, deterministic)
# =========================================================================
BAD_RANGES  = [(497, 580)] # Set manual ranges of bad data
FIT_POINTS  = 8                
STITCH_COLS = [
    "SOH_pct", "Discharge_Capacity_Ah", "Charge_Capacity_Ah",
    "Discharge_Energy_Wh", "Charge_Energy_Wh", "dV_dt_max_V_s", "DCIR_Ohms",
]

df_out = df_out.sort_values("Cycle_Index").reset_index(drop=True)
pos  = np.arange(1, len(df_out) + 1)          # plot x-axis numbering
cols = [c for c in STITCH_COLS if c in df_out.columns]

print("\nRaw measured SOH around the bad ranges (check these look as expected):")
for a, b in BAD_RANGES:
    show = df_out.loc[(pos >= a - 5) & (pos <= b + 5), ["Cycle_Index", "SOH_pct"]].dropna()
    print(f"  range {a}-{b}:\n", show.round(3).to_string())

for a, b in sorted(BAD_RANGES):
    pre_mask, post_mask = pos < a, pos > b
    df_out.loc[(pos >= a) & (pos <= b), cols] = np.nan      # delete bad data
    x_eval = b + 1                                           # compare trends
    for col in cols:
        pre  = df_out.loc[pre_mask,  col].dropna().tail(FIT_POINTS)
        post = df_out.loc[post_mask, col].dropna().head(FIT_POINTS)
        if len(pre) < 2 or len(post) < 2:
            continue
        pre_val  = np.polyval(np.polyfit(pre.index + 1,  pre.values,  1), x_eval)
        post_val = np.polyval(np.polyfit(post.index + 1, post.values, 1), x_eval)
        offset = pre_val - post_val
        df_out.loc[post_mask, col] = df_out.loc[post_mask, col] + offset
        if col == "SOH_pct":
            print(f" -> range {a}-{b}: SOH after the range shifted by {offset:+.3f} %")

df_out[cols] = df_out[cols].interpolate(method="linear", limit_area="inside")


# =========================================================================
# 3c. AUTOMATIC REMOVAL OF RESIDUAL STEPS IN SOH
# =========================================================================
STEP_WIN_CYCLES   = 60     # local reference rate = median slope within +-60 cycles
STEP_K            = 4.0    # lower = more aggressive (try 3), higher = gentler (try 6)
STEP_REL_FLOOR    = 1.0    # a slope must differ from the local rate by at least this x the median fade rate
PROTECT_FIRST     = 40     # never touch the genuine fast early fade (first 40 rows)
ENFORCE_MONOTONIC = True   # SOH may not go back up

def remove_residual_steps(df, col="SOH_pct"):
    s = df[["Cycle_Index", col]].dropna()
    if len(s) < 10:
        return df[col], 0
    pos = (s.index.to_numpy() + 1).astype(float)      # plot x-axis numbering
    y   = s[col].to_numpy(float)
    dx  = np.diff(pos)
    slope = np.diff(y) / dx
    xm  = (pos[:-1] + pos[1:]) / 2
    ref = np.array([np.median(slope[np.abs(xm - xi) <= STEP_WIN_CYCLES]) for xi in xm])
    mad = 1.4826 * np.median(np.abs(slope - np.median(slope)))
    thr = max(STEP_K * mad, STEP_REL_FLOOR * abs(np.median(slope)), 1e-12)
    bad = (np.abs(slope - ref) > thr) & (xm > PROTECT_FIRST)
    fixed = np.where(bad, ref, slope)
    y_new = y[0] + np.concatenate([[0.0], np.cumsum(fixed * dx)])
    if ENFORCE_MONOTONIC:
        y_new = np.minimum.accumulate(y_new)
    out = df[col].copy()
    out.loc[s.index] = y_new
    return out, int(bad.sum())

df_out["SOH_pct"], n_steps = remove_residual_steps(df_out, "SOH_pct")
print(f" -> automatic pass: {n_steps} steep SOH slope(s) replaced by the local fade rate")

# =========================================================================
# 6. FINAL CLEANUP & EXPORT
# =========================================================================
df_out = df_out.ffill().bfill()
df_out.replace([np.inf, -np.inf], np.nan, inplace=True)  # If a infinity splidded trough (division by 0) then replace with 0
df_out.fillna(0.0, inplace=True)

df_out = df_out[[col for col in FEATURES_TO_EXPORT if col in df_out.columns]] # Filter the final dataframe to only include columns that were marked for inclusion

print(f"\nWriting features into: {os.path.basename(output_excel_path)}...") # Saving the final list directly to an excel file in same folder
df_out.to_excel(output_excel_path, index=False, sheet_name=f"{output_cell_name}_LSTM")
print(f"Export Completed! Matrix generated with {len(df_out.columns)} selected features.")