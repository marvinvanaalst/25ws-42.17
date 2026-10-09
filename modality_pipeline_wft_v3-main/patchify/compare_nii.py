import os
import torch
import nibabel as nib
import numpy as np
from monai.metrics import SSIMMetric
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from tqdm import tqdm
# Paths
folder1 = "/Drive4T/inam/MRMRData/T1/3T_t1"
folder2 = "/Drive4T/inam/3T_7T_Synthesis/patchify/kuku"

# SSIM Metric init
# ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0).to("cuda" if torch.cuda.is_available() else "cpu")
ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0)

# device = next(ssim_metric.parameters(), torch.tensor([])).device
device = "cuda:1"
results = []

loppy = tqdm(os.listdir(folder1))
# Iterate over all files in folder1
for idx, filename in enumerate(loppy):
    if not filename.endswith(".nii") and not filename.endswith(".nii.gz"):
        continue

    file1 = os.path.join(folder1, filename)
    print("Processing file:", file1)
    file2 = os.path.join(folder2, filename)
    print("Processing file:", file2)
    

    # Check for file existence
    if not os.path.exists(file2):
        print(f"[ERROR] Missing corresponding file in folder2: {filename}")
        continue

    # Load .nii.gz files
    img1 = nib.load(file1).get_fdata()
    img2 = nib.load(file2).get_fdata()

    # Check shape match
    if img1.shape != img2.shape:
        print(f"[ERROR] Shape mismatch for {filename}: {img1.shape} vs {img2.shape}")
        continue

    # Check value range difference
    range1 = img1.max() - img1.min()
    range2 = img2.max() - img2.min()

    # if not np.isclose(range1, range2, rtol=0.0001):  # 10% tolerance
    #     print(f"[WARNING] Intensity range differs in {file1}: "
    #           f"range1={range1:.2f}, range2={range2:.2f}")

    # Normalize both to [0, 1]
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    img1 = normalize(img1)
    img2 = normalize(img2)

    # Convert to tensors: (B, C, H, W, D)
    t1 = torch.tensor(img1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    t2 = torch.tensor(img2, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Compute SSIM (3D)
    with torch.no_grad():
        ssim_val = ssim_metric(t1, t2).item()

    # Compute PSNR
    psnr_val = compute_psnr(img1, img2, data_range=1.0)

    print(f"{filename}: PSNR = {psnr_val:.2f} dB, SSIM = {ssim_val:.4f}")
    results.append((filename, psnr_val, ssim_val))
    
        # if not np.isclose(range1, range2, rtol=0.0001):  # 10% tolerance
    #     print(f"[WARNING] Intensity range differs in {file1}: "
    #           f"range1={range1:.2f}, range2={range2:.2f}")

# Summary
if results:
    mean_psnr = np.mean([r[1] for r in results])
    mean_ssim = np.mean([r[2] for r in results])
    print(f"\n✅ Summary: Mean PSNR = {mean_psnr:.2f} dB, Mean SSIM = {mean_ssim:.4f}")
else:
    print("\n❌ No valid file pairs compared.")
