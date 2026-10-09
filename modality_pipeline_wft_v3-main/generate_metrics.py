import pandas as pd
from pathlib import Path

# Path to your logs directory
logs_dir = Path("logs")

# Prepare a list to store summary rows
summary_rows = []

# Loop over each experiment folder in logs/
for exp_dir in logs_dir.iterdir():
    if not exp_dir.is_dir():
        continue  # skip files
    
    print(f"Checking experiment: {exp_dir.name}")
    
    metrics_dir = exp_dir / "metrics"
    if not metrics_dir.exists():
        print(f"  ❌ Metrics folder not found: {metrics_dir}")
        continue  # skip if metrics folder doesn't exist
    
    # Find *_testing_.csv files
    csv_files = list(metrics_dir.glob("*_testing_.csv"))
    if not csv_files:
        print(f"  ⚠️ No *_testing_.csv files found in: {metrics_dir}")
        continue  # skip if no files found
    
    print(f"  ✅ Found {len(csv_files)} testing CSV file(s)")
    
    # Read and combine all CSVs
    dfs = [pd.read_csv(f) for f in csv_files]
    combined = pd.concat(dfs, ignore_index=True)
    
    # Keep only numeric columns for averaging
    numeric_cols = combined.select_dtypes(include="number").columns
    
    # Compute averages
    averages = combined[numeric_cols].mean()
    
    # Create a summary row
    row = {
        "Experiment name": exp_dir.name,
        "number_of_folds": len(csv_files)
    }
    row.update(averages.to_dict())
    
    summary_rows.append(row)

# Create summary DataFrame
summary_df = pd.DataFrame(summary_rows)

# Save as CSV in the logs directory
output_path = logs_dir / "experiments_summary_16oct.csv"
summary_df.to_csv(output_path, index=False)

print(f"\nSummary saved to {output_path}")
