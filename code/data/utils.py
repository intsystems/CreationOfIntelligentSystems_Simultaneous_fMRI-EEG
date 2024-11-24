import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import torch
from skimage import morphology, filters

def show_slices(slices):
    _, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        axes[i].axis('off')
    plt.show()


def show_slices_with_mask(fmri_tensor, mask_tensor):
    # Check that the dimensions of the tensors match
    assert fmri_tensor.shape == mask_tensor.shape, "The dimensions of the fMRI and mask tensors must match"

    # Convert tensors to numpy arrays
    fmri_array = fmri_tensor.numpy()
    mask_array = mask_tensor.numpy()

    # Normalize fMRI data to the range [0, 1]
    fmri_array = (fmri_array - fmri_array.min()) / (fmri_array.max() - fmri_array.min())

    num_slices = 3
    dim_slices = (fmri_array.shape[0]//2, fmri_array.shape[1]//2, fmri_array.shape[2]//2)
    fig, axes = plt.subplots(1, num_slices, figsize=(15, 5))
    cmap = colors.ListedColormap(['black', 'red'])
    for i in range(num_slices):
        if i == 0:
            axes[i].imshow(fmri_array[dim_slices[i], :, :].T, cmap="gray", origin="lower")
            mask_overlay = np.ma.masked_where(mask_array[dim_slices[i], :, :] == 0, mask_array[dim_slices[i], :, :])
            axes[i].imshow(mask_overlay.T, cmap=cmap, alpha=0.4, origin="lower", vmin=0, vmax=1)
            axes[i].axis('off')
        elif i == 1:
            axes[i].imshow(fmri_array[:, dim_slices[i], :].T, cmap="gray", origin="lower")
            mask_overlay = np.ma.masked_where(mask_array[:, dim_slices[i], :] == 0, mask_array[:, dim_slices[i], :])
            axes[i].imshow(mask_overlay.T, cmap=cmap, alpha=0.4, origin="lower", vmin=0, vmax=1)
            axes[i].axis('off')
        else:
            axes[i].imshow(fmri_array[:, :, dim_slices[i]].T, cmap="gray", origin="lower")
            mask_overlay = np.ma.masked_where(mask_array[:, :, dim_slices[i]] == 0, mask_array[:, :, dim_slices[i]])
            axes[i].imshow(mask_overlay.T, cmap=cmap, alpha=0.4, origin="lower", vmin=0, vmax=1)
            axes[i].axis('off')

    plt.show()


def get_thresholded_change_mask(X, threshold_method=filters.threshold_otsu):
    # Calculate the absolute changes in voxels over the time series
    absolute_changes = torch.abs(X[..., 1:] - X[..., :-1])
    summed_changes = torch.sum(absolute_changes, dim=-1)
    summed_changes = (summed_changes - summed_changes.min()) / (summed_changes.max() - summed_changes.min())
    summed_changes_np = summed_changes.numpy()
    # Apply Otsu's method to automatically select the threshold
    threshold = threshold_method(summed_changes_np)
    binary_mask = summed_changes_np > threshold
    # Apply morphological operations to remove noise and fill holes in the mask
    brain_mask = morphology.binary_closing(binary_mask, morphology.ball(2))
    brain_mask = morphology.binary_opening(brain_mask, morphology.ball(2)).astype(np.int8)

    return torch.tensor(brain_mask)