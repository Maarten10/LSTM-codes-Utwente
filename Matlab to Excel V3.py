# =========================================================================
# Title: Nasa Matlab data to Excel converter
# Version: 3.0
# Date: 28-09-2026
# Last rev: 06-10-2026
# Added in last rev: Process all cells simultaeously 
# Organisation: University of Twente, Electrical Engineering, Power Electronics
# Author: Maarten Mulder
# Description: Converts the Matlab files of the NASA dataset into an Excel 
#              input features file. Set the file directory of the folder these 
#              files are in and set the metadata per cell and it is set up.
# =========================================================================

import os
import sys
import numpy as np
import scipy.io
import pandas as pd

# =========================================================================
# 1. USER INPUT PARAMETERS & BATCH CONFIGURATION
# =========================================================================
data_dir = r"C:\Users\mjjmu\OneDrive\Bureaublad\LSTM Testmap"

# Dictionary containing the precise physical metadata for every cell group
CELL_CONFIGS = {
    # Group 1: Room Temp, 2A Constant Discharge, 2.7V Cut-off
    "B0005": {"C_rate": 1.0, "Temperature": 24.0, "Cut_off": 2.7, "Nominal_Capacity": 2.0},
    "B0006": {"C_rate": 1.0, "Temperature": 24.0, "Cut_off": 2.5, "Nominal_Capacity": 2.0},
    "B0007": {"C_rate": 1.0, "Temperature": 24.0, "Cut_off": 2.2, "Nominal_Capacity": 2.0},
    "B0018": {"C_rate": 1.0, "Temperature": 24.0, "Cut_off": 2.5, "Nominal_Capacity": 2.0},
    
    # Group 4: Room Temp, 4A Constant Discharge (High Current)
    "B0033": {"C_rate": 2.0, "Temperature": 24.0, "Cut_off": 2.0, "Nominal_Capacity": 2.0},
    "B0034": {"C_rate": 2.0, "Temperature": 24.0, "Cut_off": 2.2, "Nominal_Capacity": 2.0},
    
    # Group 3: Elevated Temp (43°C), 4A Constant Discharge
    "B0029": {"C_rate": 2.0, "Temperature": 43.0, "Cut_off": 2.0, "Nominal_Capacity": 2.0},
    "B0030": {"C_rate": 2.0, "Temperature": 43.0, "Cut_off": 2.2, "Nominal_Capacity": 2.0},
    "B0031": {"C_rate": 2.0, "Temperature": 43.0, "Cut_off": 2.5, "Nominal_Capacity": 2.0},
    "B0032": {"C_rate": 2.0, "Temperature": 43.0, "Cut_off": 2.7, "Nominal_Capacity": 2.0},
    
    # Group 7 & 9: Cold Temp (4°C), 1A Constant Discharge
    "B0053": {"C_rate": 1.0, "Temperature": 4.0, "Cut_off": 2.0, "Nominal_Capacity": 2.0},
    "B0054": {"C_rate": 1.0, "Temperature": 4.0, "Cut_off": 2.2, "Nominal_Capacity": 2.0},
    "B0055": {"C_rate": 1.0, "Temperature": 4.0, "Cut_off": 2.5, "Nominal_Capacity": 2.0},
    "B0056": {"C_rate": 1.0, "Temperature": 4.0, "Cut_off": 2.7, "Nominal_Capacity": 2.0},
}

# =========================================================================
# 2. BATCH PROCESSING LOOP
# =========================================================================
for target_cell, cell_meta in CELL_CONFIGS.items():
    print("=" * 60)
    print(f"Processing target cell: {target_cell}")
    
    full_mat_path = os.path.join(data_dir, f"{target_cell}.mat")
    if not os.path.exists(full_mat_path):
        print(f"WARNING: Could not find file {full_mat_path}. Skipping...")
        continue

    mat = scipy.io.loadmat(full_mat_path, struct_as_record=False, squeeze_me=True)
    cycle_data = mat[target_cell].cycle if target_cell in mat else mat[list(mat.keys())[2]].cycle

    structured_rows = []
    cumulative_discharge_ah = 0.0
    cycle_count = 0

# =========================================================================
# 3. EXTRACT DATA PER OPERATION TYPE
# =========================================================================
    for op in cycle_data:
        op_type = str(op.type)
        data_struct = op.data
        
        if op_type == "discharge":
            capacity = getattr(data_struct, 'Capacity', None)
            v_meas = getattr(data_struct, 'Voltage_measured', np.array([]))
            i_meas = getattr(data_struct, 'Current_measured', np.array([]))
            
            if capacity is not None and capacity > 0 and len(v_meas) > 0:
                cycle_count += 1
                cumulative_discharge_ah += capacity
                current_efc = cumulative_discharge_ah / cell_meta["Nominal_Capacity"]
                
                # Voltage-based reference validation (with 0.1V bounce tolerance)
                min_voltage_reached = np.min(v_meas)
                is_reference = min_voltage_reached <= (cell_meta["Cut_off"] + 0.1)
                
                current_soh = (capacity / cell_meta["Nominal_Capacity"]) * 100.0 if is_reference else np.nan
                
                # DCIR Calculation
                last_known_dcir = np.nan
                if len(v_meas) > 5 and len(i_meas) > 5:
                    v_rest, i_rest = np.mean(v_meas[0:2]), np.mean(i_meas[0:2])
                    load_index = next((idx for idx, curr in enumerate(i_meas[:10]) if curr < -1.5), None)
                    if load_index is not None:
                        last_known_dcir = abs(v_rest - v_meas[load_index]) / abs(i_rest - i_meas[load_index])
                
                structured_rows.append({
                    "Cycle_Index": cycle_count,
                    "EFC": current_efc,
                    "SOH_pct": current_soh,
                    "Discharge_Capacity_Ah": capacity,
                    "DCIR_Ohms": last_known_dcir,
                    "Re_Ohms": np.nan,
                    "Rct_Ohms": np.nan,
                    "C_rate": float(cell_meta["C_rate"]),
                    "Temperature": float(cell_meta["Temperature"]),
                    "Cut_off": float(cell_meta["Cut_off"]),
                    "Nominal_Capacity": float(cell_meta["Nominal_Capacity"]),
                })

        elif op_type == "impedance":
            if structured_rows:
                structured_rows[-1]["Re_Ohms"] = getattr(data_struct, 'Re', np.nan)
                structured_rows[-1]["Rct_Ohms"] = getattr(data_struct, 'Rct', np.nan)

# =========================================================================
# 4. PANDAS FLATTENING & NORMALIZATION
# =========================================================================
    df = pd.DataFrame(structured_rows)
    if df.empty:
        print(f"WARNING: No valid discharge data found for {target_cell}. Skipping...")
        continue

    if "EFC" in df.columns:
        df["EFC"] = df["EFC"].replace(0.0, np.nan)
        if df["EFC"].isna().iloc[0]:
            df.loc[0, "EFC"] = 0.0
        df["EFC"] = df["EFC"].interpolate().ffill().bfill()

    # SOH Post-processing & Smoothing
    df["SOH_pct"] = df["SOH_pct"].interpolate().ffill().bfill()
    df['SOH_pct'] = df['SOH_pct'].rolling(window=5, min_periods=1, center=True).mean()

    # Constrained peak search (first 15 cycles) to handle warm-up stabilization safely
    search_limit = min(15, len(df))
    peak_idx = df['SOH_pct'].iloc[:search_limit].idxmax()
    true_max_soh = df.loc[peak_idx, 'SOH_pct']

    if pd.notna(peak_idx):
        df.loc[:peak_idx, 'SOH_pct'] = true_max_soh

    df['SOH_pct'] = (df['SOH_pct'] / true_max_soh) * 100.0
    df = df.ffill().bfill()

    # Export to Excel
    target_column_order = [
        "Cycle_Index", "EFC", "SOH_pct", "Discharge_Capacity_Ah", 
        "DCIR_Ohms", "Re_Ohms", "Rct_Ohms", 
        "C_rate", "Temperature", "Cut_off", "Nominal_Capacity"
    ]
    df = df[target_column_order]

    output_excel_path = os.path.join(data_dir, f"{target_cell}_LSTM_Features.xlsx")
    df.to_excel(output_excel_path, index=False, sheet_name="LSTM Input Features")
    print(f"Success! Saved: {os.path.basename(output_excel_path)} ({len(df)} rows)")

print("=" * 60)
print("Batch processing complete for all designated cells!")