from monai.transforms import MapTransform
import numpy as np

class AutoScaleToZeroOneD(MapTransform):
    def __init__(self, keys, clip=True):
        super().__init__(keys)
        self.clip = clip

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            # Get original dtype
            original_dtype = img.dtype

            # Decide float dtype (for safety, upgrade ints)
            if np.issubdtype(original_dtype, np.integer):
                float_dtype = np.float32  # or np.float64 if you prefer
            else:
                float_dtype = np.dtype(original_dtype).type  # preserve float32/float64

            # Convert to float for scaling
            img = img.astype(float_dtype)

            min_val = img.min()
            max_val = img.max()
            scaled = (img - min_val) / (max_val - min_val + 1e-8)

            if self.clip:
                scaled = np.clip(scaled, 0.0, 1.0)

            # Store back with float dtype (already ensured above)
            d[key] = scaled

        return d
