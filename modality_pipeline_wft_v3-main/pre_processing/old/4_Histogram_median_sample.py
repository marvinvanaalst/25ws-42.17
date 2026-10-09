import os
import nibabel as nib
import numpy as np
from scipy.spatial.distance import cdist

def compute_histogram(image_path, bins=100, mask=None):
    data = nib.load(image_path).get_fdata()
    if mask is not None:
        data = data[mask > 0]
    hist, _ = np.histogram(data, bins=bins, range=(np.percentile(data, 1), np.percentile(data, 99)), density=True)
    return hist

def find_median_image(folder_path, bins=100):
    image_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".nii") or f.endswith(".nii.gz")])

    print(f"Found {len(image_paths)} images.")
    histograms = [compute_histogram(p, bins=bins) for p in image_paths]
    hist_matrix = np.stack(histograms)

    # Compute pairwise distances between all histograms
    dist_matrix = cdist(hist_matrix, hist_matrix, metric='euclidean')

    # Average distance to all others
    mean_dists = dist_matrix.mean(axis=1)

    # Index of the "median" image
    median_idx = np.argmin(mean_dists)
    median_path = image_paths[median_idx]
    print(f"Median image is: {median_path}")
    return median_path

if __name__ == "__main__":
    folder_path = "path/to/your/nifti/images"  # Change this to your folder path
    median_image_path = find_median_image(folder_path, bins=100)
    print(f"Median image path: {median_image_path}")