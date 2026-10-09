import os
import pandas as pd

# Root folder containing all experiment folders
LOGS_DIR = "logs"
OUTPUT_FILE_DETAILED = "isbi_fin_1720.csv"
OUTPUT_FILE_SUMMARY = "ZZZ_summary_metrics.csv"

# Collect rows for the final CSVs
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
            
            # Save values for averaging + std later
            split_metrics.append({
                "psnr": row["test_psnr"],
                "ssim_2D": row["test_ssim_2D"],
                "ssim_3D": row["test_ssim_3D"],
                "mae": row["test_mae"],
                "mse": row["test_mse"]
            })
    
    # Compute averages and std if we got splits
    if split_metrics:
        df_metrics = pd.DataFrame(split_metrics)
        
        # Mean
        exp_data["Average_PSNR"] = df_metrics["psnr"].mean()
        exp_data["Average_SSIM_2D"] = df_metrics["ssim_2D"].mean()
        exp_data["Average_SSIM_3D"] = df_metrics["ssim_3D"].mean()
        exp_data["Average_MAE"] = df_metrics["mae"].mean()
        exp_data["Average_MSE"] = df_metrics["mse"].mean()
        
        # Std
        exp_data["Std_PSNR"] = df_metrics["psnr"].std(ddof=1)
        exp_data["Std_SSIM_2D"] = df_metrics["ssim_2D"].std(ddof=1)
        exp_data["Std_SSIM_3D"] = df_metrics["ssim_3D"].std(ddof=1)
        exp_data["Std_MAE"] = df_metrics["mae"].std(ddof=1)
        exp_data["Std_MSE"] = df_metrics["mse"].std(ddof=1)
    
    all_experiments.append(exp_data)

# Save detailed CSV (split-level + averages + stds)
df_all = pd.DataFrame(all_experiments)
df_all.to_csv(OUTPUT_FILE_DETAILED, index=False)

# Save summary CSV (only experiment, weight_prefix, averages, stds)
summary_cols = ["Experiment", "weight_prefix",
                "Average_PSNR", "Std_PSNR",
                "Average_SSIM_2D", "Std_SSIM_2D",
                "Average_SSIM_3D", "Std_SSIM_3D",
                "Average_MAE", "Std_MAE",
                "Average_MSE", "Std_MSE"]

df_summary = df_all[summary_cols]
df_summary.to_csv(OUTPUT_FILE_SUMMARY, index=False)

print(f"Saved detailed metrics to {OUTPUT_FILE_DETAILED}")
print(f"Saved summary metrics to {OUTPUT_FILE_SUMMARY}")
