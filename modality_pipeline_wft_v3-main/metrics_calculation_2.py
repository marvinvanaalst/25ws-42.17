import os
import pandas as pd

# Root folder containing all experiment folders
LOGS_DIR = "logs"
OUTPUT_FILE = "all_metrics11.csv"

# Collect rows for the final CSV
all_experiments = []

# Walk through each experiment
for exp_name in os.listdir(LOGS_DIR):
    exp_path = os.path.join(LOGS_DIR, exp_name)
    metrics_path = os.path.join(exp_path, "metrics")
    
    if not os.path.isdir(metrics_path):
        continue  # skip if no metrics folder
    
    exp_data = {"Experiment": exp_name}
    split_metrics = []
    
    # Iterate over all testing csv files in metrics folder
    for file in os.listdir(metrics_path):
        if file.endswith("testing_.csv"):
            split_idx = file.split("_")[0]  # "0" from "0_metrics_testing_.csv"
            csv_path = os.path.join(metrics_path, file)
            
            # Load CSV (single row)
            df = pd.read_csv(csv_path)
            row = df.iloc[0]
            
            # Store weight prefix once (assume consistent across splits)
            exp_data["weight_prefix"] = row["weight_prefix"]
            
            # Add metrics for this split
            exp_data[f"split_{split_idx}_PSNR"] = row["test_psnr"]
            exp_data[f"split_{split_idx}_SSIM_2D"] = row["test_ssim_2D"]
            exp_data[f"split_{split_idx}_SSIM_3D"] = row["test_ssim_3D"]
            exp_data[f"split_{split_idx}_MAE"] = row["test_mae"]
            exp_data[f"split_{split_idx}_MSE"] = row["test_mse"]
            
            # Save values for averaging later
            split_metrics.append({
                "psnr": row["test_psnr"],
                "ssim_2D": row["test_ssim_2D"],
                "ssim_3D": row["test_ssim_3D"],
                "mae": row["test_mae"],
                "mse": row["test_mse"]
            })
    
    # Compute averages if we got splits
    if split_metrics:
        avg_psnr = sum(m["psnr"] for m in split_metrics) / len(split_metrics)
        avg_ssim2d = sum(m["ssim_2D"] for m in split_metrics) / len(split_metrics)
        avg_ssim3d = sum(m["ssim_3D"] for m in split_metrics) / len(split_metrics)
        avg_mae = sum(m["mae"] for m in split_metrics) / len(split_metrics)
        avg_mse = sum(m["mse"] for m in split_metrics) / len(split_metrics)
        
        exp_data["Average_PSNR"] = avg_psnr
        exp_data["Average_SSIM_2D"] = avg_ssim2d
        exp_data["Average_SSIM_3D"] = avg_ssim3d
        exp_data["Average_MAE"] = avg_mae
        exp_data["Average_MSE"] = avg_mse
    
    all_experiments.append(exp_data)

# Save everything into one CSV
df_all = pd.DataFrame(all_experiments)
df_all.to_csv(OUTPUT_FILE, index=False)

print(f"Saved combined metrics to {OUTPUT_FILE}")
