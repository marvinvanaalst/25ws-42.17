import numpy as np

from monai.data import PatchDataset, DataLoader
from monai.transforms import RandSpatialCropSamples, RandShiftIntensity

# image dataset
images = [np.arange(16, dtype=float).reshape(1, 4, 4),
          np.arange(16, dtype=float).reshape(1, 4, 4)]
# image patch sampler
n_samples = 5
sampler = RandSpatialCropSamples(roi_size=(3, 3), num_samples=n_samples,
                                 random_center=True, random_size=False)
# patch-level intensity shifts
patch_intensity = RandShiftIntensity(offsets=1.0, prob=1.0)
# construct the patch dataset
ds = PatchDataset(dataset=images,
                  patch_func=sampler,
                  samples_per_image=n_samples,
                  transform=patch_intensity)

# use the patch dataset, length: len(images) x samplers_per_image
print(len(ds))

# >>> 10

for item in DataLoader(ds, batch_size=2, shuffle=True, num_workers=2):
    print(item.shape)

# >>> torch.Size([2, 1, 3, 3])