import os
import pandas as pd
from collections import defaultdict

LOGS_DIR = "logs"
OUTPUT_DIR = "dataset_metrics_outputs_long"
os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset_results = defaultdict(list)

for exp_name in os.listdir(LOGS_DIR):
    exp_path = os.path.join(LOGS_DIR, exp_name)
    datasetEvals_path = os.path.join(exp_path, "datasetEvals")
    
    if not os.path.isdir(datasetEvals_path):
        continue
    
    for dataset_name in os.listdir(datasetEvals_path):
        dataset_path = os.path.join(datasetEvals_path, dataset_name)
        metrics_path = os.path.join(dataset_path, "metrics")
        
        if not os.path.isdir(metrics_path):
            continue
        
        split_rows = []
        
        for file in os.listdir(metrics_path):
            if file.endswith("testing_.csv"):
                split_idx = file.split("_")[0]
                csv_path = os.path.join(metrics_path, file)
                
                df = pd.read_csv(csv_path)
                row = df.iloc[0]
                
                split_rows.append({
                    "Experiment": exp_name,
                    "weight_prefix": row["weight_prefix"],
                    "Split": split_idx,
                    "PSNR": row["test_psnr"],
                    "SSIM_2D": row["test_ssim_2D"],
                    "SSIM_3D": row["test_ssim_3D"],
                    "MAE": row["test_mae"],
                    "MSE": row["test_mse"]
                })
        
        if split_rows:
            df_splits = pd.DataFrame(split_rows)
            n_splits = len(df_splits)
            
            # Compute averages across splits
            avg_row = {
                "Experiment": exp_name,
                "weight_prefix": df_splits["weight_prefix"].iloc[0],
                "Split": f"AverageOf {n_splits} Splits",
                "PSNR": df_splits["PSNR"].mean(),
                "SSIM_2D": df_splits["SSIM_2D"].mean(),
                "SSIM_3D": df_splits["SSIM_3D"].mean(),
                "MAE": df_splits["MAE"].mean(),
                "MSE": df_splits["MSE"].mean()
            }
            
            # Add std row too (optional)
            std_row = {
                "Experiment": exp_name,
                "weight_prefix": df_splits["weight_prefix"].iloc[0],
                "Split": f"StdOf {n_splits} Splits",
                "PSNR": df_splits["PSNR"].std(ddof=1),
                "SSIM_2D": df_splits["SSIM_2D"].std(ddof=1),
                "SSIM_3D": df_splits["SSIM_3D"].std(ddof=1),
                "MAE": df_splits["MAE"].std(ddof=1),
                "MSE": df_splits["MSE"].std(ddof=1)
            }
            
            dataset_results[dataset_name].extend(split_rows + [avg_row, std_row])

# Save one long-format CSV per dataset
for dataset_name, records in dataset_results.items():
    df_out = pd.DataFrame(records)
    output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_long_metrics.csv")
    df_out.to_csv(output_path, index=False)
    print(f"Saved long-format metrics for {dataset_name} -> {output_path}")
