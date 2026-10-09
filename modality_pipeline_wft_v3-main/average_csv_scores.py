import pandas as pd
from pathlib import Path

# Set directory containing the CSV files
folder = Path("/storage/an_inam/CT2MR/CustomPipelineV2/logs/2D_ESAU_L1LPIPS_608_256_finetune_masked_full_lr_low_b/metrics")  

# Find all files matching *_testing_.csv
csv_files = list(folder.glob("*_testing_.csv"))

# Read all and collect metrics
dfs = [pd.read_csv(f) for f in csv_files]

# Concatenate into one big dataframe
combined = pd.concat(dfs, ignore_index=True)

# Select metric columns
metrics = ["test_psnr", "test_ssim", "test_mae", "test_mse"]
averages = combined[metrics].mean()

# Save to new CSV in the same directory
output_path = folder / "average.csv"
averages.to_frame().T.to_csv(output_path, index=False)

print(f"Averages saved to {output_path}")
