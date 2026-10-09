import SimpleITK as sitk
import numpy as np


def mi_contribution_map(
        fixed_path,
        moving_path,
        bins=32,
        radius=2,
        sample_percent=1.0  # full sampling
    ):
    # Load images
    fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)

    arr = sitk.GetArrayFromImage(fixed)  # shape: (Z,Y,X)
    Z, Y, X = arr.shape
    r = radius

    # Identity transform → assume aligned
    transform = sitk.Transform(3, sitk.sitkIdentity)

    # Setup registration to evaluate metric
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bins)
    reg.SetMetricSamplingStrategy(reg.NONE)  # full voxel sampling

    reg.SetFixedImage(fixed)
    reg.SetMovingImage(moving)
    reg.SetInitialTransform(transform)

    # Baseline global MI
    I_global = reg.GetMetricValue()

    mi_vol = np.zeros((Z, Y, X), dtype=np.float32)

    # Loop spatially
    for z in range(r, Z - r):
        for y in range(r, Y - r):
            for x in range(r, X - r):
                
                # Create local exclusion mask
                mask = np.ones((Z, Y, X), dtype=np.uint8)
                mask[z-r:z+r+1, y-r:y+r+1, x-r:x+r+1] = 0

                mask_img = sitk.GetImageFromArray(mask)
                mask_img.CopyInformation(fixed)

                # Apply masking to both images
                reg.SetMetricFixedMask(mask_img)
                reg.SetMetricMovingMask(mask_img)

                I_minus = reg.GetMetricValue()
                mi_vol[z, y, x] = I_global - I_minus

    return mi_vol, fixed


s1 = "/storage/an_inam/datasets/CT_MR/pelvis_compliant/ct/1PC000.nii.gz"
s2 = "/storage/an_inam/datasets/CT_MR/pelvis_compliant/mr/1PC000.nii.gz"


mi_vol, ref_img = mi_contribution_map(s1, s2)

out = sitk.GetImageFromArray(mi_vol)
out.CopyInformation(ref_img)
sitk.WriteImage(out, "local_MI_map.nii.gz")
print("Saved: local_MI_map.nii.gz")

# mi_map = mi_contribution_map(s1, s2, bins=32, radius=2)
# sitk.WriteImage(sitk.GetImageFromArray(mi_map), "mi_heatmap.nii.gz")