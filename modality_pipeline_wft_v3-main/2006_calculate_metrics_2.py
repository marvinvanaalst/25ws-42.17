import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

def process_experiment_results(parent_dir, destination_dir, id_token):
    parent_path = Path(parent_dir)
    dest_path = Path(destination_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    metrics_cols = ['test_psnr', 'test_ssim_3D', 'test_ssim_2D', 'test_mae', 'test_mse']
    
    all_experiments_data = []
    compiled_results = []
    error_log = []

    # Iterate through each folder (experiment)
    for exp_folder in parent_path.iterdir():
        if not exp_folder.is_dir():
            continue
        
        log_dir = exp_folder / "metrics"
        if not log_dir.exists():
            error_log.append(f"Directory Error: Missing 'log' folder in {exp_folder.name}")
            continue

        # Find all cross-fold CSVs
        csv_files = list(log_dir.glob("*_metircs_testing_.csv"))
        file_count = len(csv_files)
        
        if file_count == 0:
            error_log.append(f"File Error: No '*_metircs_testing_.csv' files found in {log_dir}")
            continue

        fold_data_list = []
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                if df.empty:
                    error_log.append(f"Data Error: {f.name} is empty in {exp_folder.name}")
                    continue
                fold_data_list.append(df)
            except Exception as e:
                error_log.append(f"Read Error: Could not read {f.name} in {exp_folder.name} - {str(e)}")

        if not fold_data_list:
            continue

        combined_folds = pd.concat(fold_data_list, ignore_index=True)
        grouped = combined_folds.groupby('weight_prefix')

        for weight_name, group in grouped:
            base_info = {
                'experiment': exp_folder.name,
                'weight_prefix': weight_name,
                'file_count': file_count
            }
            
            stats_row = base_info.copy()
            summary_row = base_info.copy()

            for col in metrics_cols:
                if col not in group.columns:
                    continue
                
                data = group[col].dropna()
                if data.empty:
                    continue
                
                mean_val = data.mean()
                std_val = data.std()
                t_stat, _ = stats.ttest_1samp(data, 0) if len(data) > 1 else (np.nan, np.nan)

                stats_row.update({
                    f'{col}_avg': mean_val,
                    f'{col}_std': std_val,
                    f'{col}_tscore': t_stat
                })
                summary_row[col] = mean_val

            compiled_results.append(stats_row)
            all_experiments_data.append(summary_row)

    # --- DataFrame Creation & Sorting ---
    df_metrics = pd.DataFrame(compiled_results)
    df_aggregate = pd.DataFrame(all_experiments_data)

    # Sort by PSNR descending (highest first)
    if not df_metrics.empty and 'test_psnr_avg' in df_metrics.columns:
        df_metrics = df_metrics.sort_values(by='test_psnr_avg', ascending=False)
    
    if not df_aggregate.empty and 'test_psnr' in df_aggregate.columns:
        df_aggregate = df_aggregate.sort_values(by='test_psnr', ascending=False)

    # --- Save Results ---
    file1_name = f"{id_token}_metrics.csv"
    file2_name = f"{id_token}_aggregate.csv"

    df_metrics.to_csv(dest_path / file1_name, index=False)
    df_aggregate.to_csv(dest_path / file2_name, index=False)

    # --- Elaborate Error Reporting ---
    print("-" * 50)
    print(f"COMPILATION COMPLETE (Sorted by PSNR)")
    print(f"1. Detailed Metrics: {file1_name}")
    print(f"2. Aggregate Model Comparison: {file2_name}")
    print("-" * 50)
    
    if error_log:
        print("\nERRORS & MISSING DATA ENCOUNTERED:")
        for error in error_log:
            print(f"  [!] {error}")
    else:
        print("\nAll folders processed successfully.")

# --- Configuration ---
SOURCE_PATH = "logs"   
DEST_PATH = "ISBI_24_JAN"         
IDENTIFIER = "VNET_L1"     

import os
os.makedirs(DEST_PATH,exist_ok=True)     

if __name__ == "__main__":
    process_experiment_results(SOURCE_PATH, DEST_PATH, IDENTIFIER)