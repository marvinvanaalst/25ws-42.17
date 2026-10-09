import os
import torch
import torchio as tio
from glob import glob
from tqdm import tqdm
from pathlib import Path

def load_patches(patch_dir):
    patch_files = sorted(glob(str(Path(patch_dir) / 'patch_*.pt')))
    if not patch_files:
        raise FileNotFoundError(f"No patch_*.pt files found in {patch_dir}")
    return patch_files


def reconstruct_volume(patch_dir, output_path):
    patch_files = load_patches(patch_dir)
    

        
    sample_patch = torch.load(patch_files[0])

    patch_size = list(sample_patch['patch'].shape[-3:])
    print(patch_size)
    affine = sample_patch['affine'].numpy()
    original_path = sample_patch['filename']
    print(original_path)

    # Dummy subject to match dimensions (not actually used for image content)
    dummy_subject = tio.Subject(image=tio.ScalarImage(original_path))
    
    aggregator = tio.GridAggregator(sampler=tio.GridSampler(dummy_subject, patch_size))

    for patch_file in tqdm(patch_files, desc=f"Reconstructing from {patch_dir}"):
        data = torch.load(patch_file)
        print(f"{data['location']}")
        
        patch = data['patch']
        print(patch.ndim)
        # if patch.ndim == 3:
        patch = patch.unsqueeze(0)
        location = data['location']
        aggregator.add_batch(patch, torch.tensor([location]))

    full_tensor = aggregator.get_output_tensor()  # shape: (1, D, H, W)

    # Save using TorchIO ScalarImage
    scalar_img = tio.ScalarImage(tensor=full_tensor, affine=affine)
    scalar_img.save(output_path)
    print(f"Saved full volume to {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python reconstruct_volume.py <patch_dir> <output_nii_path>")
    else:
        files = sorted(os.listdir(sys.argv[1]))
        scans = [os.path.join(sys.argv[1], f) for f in files]
                
        patch_dir = sys.argv[1]
        # patch_dir = '/Drive4T/inam/MRMRData/T1/patches/64X8/3T_t1/sub-01_T1w_defaced_registered'
        
        output_path = sys.argv[2]
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(sys.argv[2]):
            os.makedirs(sys.argv[2])
        
        for ele in scans:
            output_path = f"{sys.argv[2]}/{ele.split('/')[-1]}.nii.gz"
            reconstruct_volume(ele, output_path)
            
            
        
        
        # real_files = os.listdir(sys.argv[2])
        

        # reconstruct_volume(patch_dir, output_path)
