import os
import pandas as pd
from collections import defaultdict

LOGS_DIR = "logs"
OUTPUT_DIR = "dataset_metrics_outputscc"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# We'll collect results grouped by dataset name
dataset_results = defaultdict(list)

# Walk through each experiment
for exp_name in os.listdir(LOGS_DIR):
    exp_path = os.path.join(LOGS_DIR, exp_name)
    datasetEvals_path = os.path.join(exp_path, "datasetEvals")
    
    if not os.path.isdir(datasetEvals_path):
        continue  # skip if no datasetEvals folder
    
    # For each dataset inside datasetEvals
    for dataset_name in os.listdir(datasetEvals_path):
        dataset_path = os.path.join(datasetEvals_path, dataset_name)
        metrics_path = os.path.join(dataset_path, "metrics")
        
        if not os.path.isdir(metrics_path):
            continue
        
        exp_data = {"Experiment": exp_name}
        split_metrics = []
        
        for file in os.listdir(metrics_path):
            if file.endswith("testing_.csv"):
                split_idx = file.split("_")[0]  # "0" from "0_metrics_testing_.csv"
                csv_path = os.path.join(metrics_path, file)
                
                df = pd.read_csv(csv_path)
                row = df.iloc[0]
                
                # Store weight prefix
                exp_data["weight_prefix"] = row["weight_prefix"]
                
                # Add split-level metrics
                exp_data[f"split_{split_idx}_PSNR"] = row["test_psnr"]
                exp_data[f"split_{split_idx}_SSIM_2D"] = row["test_ssim_2D"]
                exp_data[f"split_{split_idx}_SSIM_3D"] = row["test_ssim_3D"]
                exp_data[f"split_{split_idx}_MAE"] = row["test_mae"]
                exp_data[f"split_{split_idx}_MSE"] = row["test_mse"]
                
                # Save for averages/std
                split_metrics.append({
                    "psnr": row["test_psnr"],
                    "ssim_2D": row["test_ssim_2D"],
                    "ssim_3D": row["test_ssim_3D"],
                    "mae": row["test_mae"],
                    "mse": row["test_mse"]
                })
        
        if split_metrics:
            df_metrics = pd.DataFrame(split_metrics)
            
            # Averages
            exp_data["Average_PSNR"] = df_metrics["psnr"].mean()
            exp_data["Average_SSIM_2D"] = df_metrics["ssim_2D"].mean()
            exp_data["Average_SSIM_3D"] = df_metrics["ssim_3D"].mean()
            exp_data["Average_MAE"] = df_metrics["mae"].mean()
            exp_data["Average_MSE"] = df_metrics["mse"].mean()
            
            # Standard deviations
            exp_data["Std_PSNR"] = df_metrics["psnr"].std(ddof=1)
            exp_data["Std_SSIM_2D"] = df_metrics["ssim_2D"].std(ddof=1)
            exp_data["Std_SSIM_3D"] = df_metrics["ssim_3D"].std(ddof=1)
            exp_data["Std_MAE"] = df_metrics["mae"].std(ddof=1)
            exp_data["Std_MSE"] = df_metrics["mse"].std(ddof=1)
        
        # Append to this dataset's results
        dataset_results[dataset_name].append(exp_data)

# Now save separate CSVs for each dataset
for dataset_name, records in dataset_results.items():
    df_all = pd.DataFrame(records)
    
    # Detailed file (all splits + averages + stds)
    detailed_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_all_metrics.csv")
    df_all.to_csv(detailed_path, index=False)
    
    # Summary file (only averages + stds)
    summary_cols = ["Experiment", "weight_prefix",
                    "Average_PSNR", "Std_PSNR",
                    "Average_SSIM_2D", "Std_SSIM_2D",
                    "Average_SSIM_3D", "Std_SSIM_3D",
                    "Average_MAE", "Std_MAE",
                    "Average_MSE", "Std_MSE"]
    
    df_summary = df_all[summary_cols]
    summary_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_summary_metrics.csv")
    df_summary.to_csv(summary_path, index=False)
    
    print(f"Saved {dataset_name} metrics -> {detailed_path}, {summary_path}")
