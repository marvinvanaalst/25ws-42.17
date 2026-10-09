import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
def n4_bias_field_correction_batch(
    input_dir,
    output_dir,
    mask_dir=None,
    suffix="_n4corrected.nii.gz"
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mask_dir:
        mask_dir = Path(mask_dir)

    nii_files = sorted(input_dir.glob("*.nii")) + sorted(input_dir.glob("*.nii.gz"))
    loopy = tqdm(nii_files)
    for isx, nii_file in enumerate(loopy):
        output_file = output_dir / (nii_file.stem + suffix)
        print(f"[PROCESSING] {nii_file.name}")

        image = sitk.ReadImage(str(nii_file), sitk.sitkFloat32)

        # Load mask if provided
        if mask_dir:
            mask_file = mask_dir / nii_file.name
            if not mask_file.exists():
                print(f"[WARNING] Mask not found: {mask_file.name}, skipping.")
                continue
            mask = sitk.ReadImage(str(mask_file), sitk.sitkUInt8)
        else:
            # Generate fallback mask using Otsu (not ideal)
            print(f"[INFO] No mask dir provided, using Otsu for {nii_file.name}")
            mask = sitk.OtsuThreshold(image, 0, 1, 200)

        # Apply N4 bias correction
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected = corrector.Execute(image, mask)

        sitk.WriteImage(corrected, str(output_file))
        print(f"[DONE] → {output_file.name}\n")

# Example usage
n4_bias_field_correction_batch(
    input_dir="/Drive4T/inam/datasets/20_patient/raws/preprocessing/1_skull_strip/T1/7T",
    output_dir="/Drive4T/inam/datasets/20_patient/raws/preprocessing/2_biasfiled_corrected/T1/7T",
    mask_dir="/Drive4T/inam/datasets/20_patient/raws/preprocessing/1_skull_strip/T1/7T/mask_7T"  # or None if you want Otsu fallback
)
