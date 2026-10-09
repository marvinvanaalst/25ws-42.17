import torch
import nibabel as nib
import numpy as np

orig_path = "logs/exp_2.5d_lsgan_composite_paper_params/generations/Split0_0_sub-08_T1w_original.nii.gz"
fake_path = "logs/exp_2.5d_lsgan_composite_paper_params/generations/Split0_0_sub-08_T1w_saveEvery_epoch_epoch=100_fake.nii.gz"

orig = nib.load(orig_path).get_fdata()
fake = nib.load(fake_path).get_fdata()

print("Orig shape:", orig.shape, "min:", orig.min(), "max:", orig.max(), "mean:", orig.mean())
print("Fake shape:", fake.shape, "min:", fake.min(), "max:", fake.max(), "mean:", fake.mean())

diff = np.abs(orig - fake)
print("Mean absolute diff:", diff.mean(), "Max diff:", diff.max())

# Compare with Exp 6
exp6_path = "logs/exp_2.5d_unet_l1_paper_params/generations/Split0_0_sub-08_T1w_saveEvery_epoch_epoch=100_fake.nii.gz"
try:
    exp6 = nib.load(exp6_path).get_fdata()
    print("Exp6 shape:", exp6.shape, "min:", exp6.min(), "max:", exp6.max(), "mean:", exp6.mean())
    print("Exp6 vs Orig diff:", np.abs(orig - exp6).mean())
except Exception as e:
    print("Exp6 load error:", e)
