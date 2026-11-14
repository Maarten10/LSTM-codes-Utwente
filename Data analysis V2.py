# =========================================================================
# Title: Cell data analysis code 
# Version: 2.0
# Date: 19-08-2026
# Last rev: 19-10-2026
# Added in last rev: Also generate and plot the Sparmann correlation factor.
# Organisation: University of Twente, Electrical Engineering, Power Electronics
# Author: Maarten Mulder
# Description: Code analysis the data from the experiments and plots all of them together.
#              Code then outputs plots via the console. The code also generates
#              The Spearman correlation values and outputs them.
# =========================================================================

import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================================
# 1. CONFIGURATION & FOLDER SETUP.
# =========================================================================
# Define the master directory where all your extracted Excel and CSV files are stored.
folder_path = r"C:\Users\mjjmu\OneDrive\Bureaublad\LSTM Testmap"

print(f"Scanning directory: {folder_path}\n")

# Create a regular expression (regex) rule to automatically find the word "Cell_" followed by numbers.
# This ensures we can dynamically extract names like "Cell_13" from messy filenames.
cell_pattern = re.compile(r'(Cell_\d+)', re.IGNORECASE)


# =========================================================================
# 2. READ ALL FILES.
# =========================================================================
# Dictionaries to store the loaded dataframes, using the cell name as the key.
feature_dict = {}
acim_dict = {}

# --- Read Excel Feature Files ---
# Construct a search pattern to find all Excel files containing "LSTM_Features".
excel_pattern = os.path.join(folder_path, "Cell_*_LSTM_Features*.xlsx")
excel_files = glob.glob(excel_pattern)

for file in excel_files:
    # Search the filename for our regex pattern. (e.g., extracting "Cell_13")
    match = cell_pattern.search(os.path.basename(file))
    if match:
        # Standardize the capitalization so "cell_13" becomes "Cell_13".
        cell_name = match.group(1).capitalize()
        print(f"Loading features for {cell_name}...")
        
        # Read the Excel file into pandas and store it in our dictionary.
        df = pd.read_excel(file)
        feature_dict[cell_name] = df

# --- Read CSV ACIM Files ---
# Construct a search pattern to find all raw ACIM (impedance) CSV files.
csv_pattern = os.path.join(folder_path, "Cell_*_*ACIM.CSV")
csv_files = glob.glob(csv_pattern)

for file in csv_files:
    # Extract the cell name just like we did for the Excel files.
    match = cell_pattern.search(os.path.basename(file))
    if match:
        cell_name = match.group(1).capitalize()
        print(f"Loading ACIM data for {cell_name}...")
        
        # Read the CSV file into pandas.
        df = pd.read_csv(file)
        # Strip trailing spaces from column headers (e.g., "Zreal " becomes "Zreal") to prevent key errors later.
        df.columns = df.columns.str.strip() 
        acim_dict[cell_name] = df

print(f"\nSuccessfully loaded {len(feature_dict)} feature files and {len(acim_dict)} ACIM files.\n")


# =========================================================================
# 3. GENERATE NYQUIST PLOT. (ALL CELLS)
# =========================================================================
if acim_dict:
    print("Generating Nyquist Plot...")
    plt.figure(figsize=(10, 8))
    
    # Loop through every loaded ACIM dataframe.
    for cell_name, df_acim in acim_dict.items():
        # Check if the real and imaginary impedance columns actually exist.
        if 'Zreal' in df_acim.columns and 'Zimg' in df_acim.columns:
            # Plot Real Impedance on the X-axis and inverted Imaginary Impedance (-Zimg) on the Y-axis.
            # Standard electrochemical convention dictates plotting -Zimg to keep the arc in the top right quadrant.
            plt.scatter(df_acim['Zreal'], -df_acim['Zimg'], label=cell_name, s=15, alpha=0.6)
            
    plt.title('Nyquist Plot Comparison (All Cells)')
    plt.xlabel("Z' (Real Impedance) [\u03a9]")
    plt.ylabel("-Z'' (Imaginary Impedance) [\u03a9]")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save the aggregated plot to disk and display it.
    nyquist_path = os.path.join(folder_path, "Nyquist_Plot_All_Cells.png")
    plt.savefig(nyquist_path, dpi=300)
    plt.show() 
else:
    print("No ACIM files found to plot.")


# =========================================================================
# 4. FEATURE DEGRADATION CORRELATION (ACTUAL VALUES)
# =========================================================================
mean_corr = pd.Series(dtype=float)
# The specific feature we want to evaluate everything against. (our ground truth proxy for SOH)
target_feature = "Discharge_Capacity_Ah"

if feature_dict:
    print("Calculating Feature Degradation Rankings (Spearman Correlation)...")
    all_correlations = []
    
    for cell_name, df in feature_dict.items():
        if target_feature in df.columns:
            # Filter out constants: Only keep columns where the value changes. (variance > 0)
            # You can't calculate correlation on a perfectly flat line.
            df_varying = df.loc[:, df.nunique() > 1]
            # Drop any rows where our target feature is NaN to prevent math errors.
            df_varying = df_varying.dropna(subset=[target_feature])
            
            # Calculate the Spearman rank correlation matrix for all varying features.
            # Spearman is used instead of Pearson because battery degradation is non-linear.
            corrmat = df_varying.corr(method='spearman')
            
            if target_feature in corrmat.columns:
                # Extract only the column comparing everything to our target_feature.
                target_corr = corrmat[target_feature].drop(target_feature, errors='ignore')
                target_corr.name = cell_name
                all_correlations.append(target_corr)
                
    if all_correlations:
        # Concatenate the correlation series from all cells side-by-side.
        df_corr = pd.concat(all_correlations, axis=1)
        # Average the correlation scores across all cells to find globally robust predictors.
        mean_corr_raw = df_corr.mean(axis=1).dropna()
        
        # Sort the features by their absolute strength (0.0 to 1.0) so the most important ones jump to the top,
        # but retain their positive/negative sign to show if they correlate directly or inversely.
        mean_corr = mean_corr_raw.reindex(mean_corr_raw.abs().sort_values(ascending=True).index)
        
        # Plotting the horizontal bar chart
        plt.figure(figsize=(12, 10))
        
        # Color-code the bars: red indicates inverse correlation (as SOH drops, this feature rises), blue indicates direct.
        colors = ['indianred' if val < 0 else 'steelblue' for val in mean_corr.values]
        bars = plt.barh(mean_corr.index, mean_corr.values, color=colors)
        
        # Draw a solid black line down the exact middle. (zero correlation)
        plt.axvline(x=0, color='black', linewidth=1.5)
        
        # Loop through the bars to dynamically place the exact number next to the bar tip.
        for bar, val in zip(bars, mean_corr.values):
            if val < 0:
                # If negative, place text slightly to the left of the bar.
                plt.text(val - 0.015, bar.get_y() + bar.get_height()/2, 
                         f'{val:.3f}', va='center', ha='right', fontsize=10)
            else:
                # If positive, place text slightly to the right of the bar.
                plt.text(val + 0.015, bar.get_y() + bar.get_height()/2, 
                         f'{val:.3f}', va='center', ha='left', fontsize=10)
        
        plt.title(f'Feature Importance: Predictors of Degradation\n(Spearman Correlation with {target_feature})', fontsize=14)
        plt.xlabel('Spearman Correlation Score (-1.0 to +1.0)', fontsize=12)
        
        # Draw visual threshold lines to indicate features with strong predictive power. (> 0.8 or < -0.8)
        plt.axvline(x=0.8, color='steelblue', linestyle='--', alpha=0.4, label='Strong Positive (> 0.8)')
        plt.axvline(x=-0.8, color='indianred', linestyle='--', alpha=0.4, label='Strong Negative (< -0.8)')
        
        # Extend the X-axis bounds slightly beyond 1.0 so the text labels don't get cut off.
        plt.xlim(-1.15, 1.15)
        plt.grid(axis='x', linestyle='--', alpha=0.4)
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        save_path = os.path.join(folder_path, "Feature_Degradation_Ranking.png")
        plt.savefig(save_path, dpi=300)
        plt.show()
        
        # Print the exact calculated rankings to the console for quick reading.
        print(f"\n--- ALL FEATURE CORRELATION RANKINGS (Target: {target_feature}) ---")
        print(mean_corr[::-1].to_string(float_format="{:.4f}".format))
    else:
         print(f"\nWarning: Could not calculate correlations. Make sure '{target_feature}' exists.")


# =========================================================================
# 5. GENERATE FEATURE GRAPHS.
# =========================================================================
if feature_dict:
    print("\nGenerating Feature Comparison Plots...")
    
    # Collect every unique column name across all loaded dataframes.
    all_features = set()
    for df in feature_dict.values():
        all_features.update(df.columns.tolist())
    
    # Generate one plot per unique feature.
    for feature in all_features:
        plt.figure(figsize=(10, 6))
        plotted_any = False
        
        # Overlay the curve for this specific feature from every cell onto the same graph.
        for cell_name, df in feature_dict.items():
            if feature in df.columns:
                # df.index + 1 acts as the "Cycle Number" on the X-axis.
                plt.plot(df.index + 1, df[feature], label=cell_name, linewidth=2)
                plotted_any = True
                
        if plotted_any:
            # Dynamically generate a subtitle indicating how strongly this feature correlates with degradation.
            if feature == target_feature:
                corr_text = "\n(Target Variable / Ground Truth)"
            elif not mean_corr.empty and feature in mean_corr:
                corr_text = f"\n(Spearman Corr: {mean_corr[feature]:.3f})"
            else:
                corr_text = "\n(Spearman Corr: N/A - Flat Line or Invalid)"
                
            plt.title(f"Comparison of {feature} Across Cells{corr_text}")
            plt.xlabel("Cycle Number")
            plt.ylabel(feature)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            
            # Clean the feature name (remove strange characters) so Windows doesn't crash when saving the file.
            safe_feature_name = "".join(x for x in feature if x.isalnum() or x in "_-")
            feature_path = os.path.join(folder_path, f"Feature_{safe_feature_name}.png")
            
            plt.savefig(feature_path, dpi=300)
            plt.show()
        else:
            # If a feature was entirely empty across all cells, delete the empty figure from memory.
            plt.close()

print("\nAll tasks completed successfully!")