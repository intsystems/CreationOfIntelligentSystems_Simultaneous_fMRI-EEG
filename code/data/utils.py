import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

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