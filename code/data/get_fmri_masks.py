import torch
import numpy as np
from skimage import morphology, filters
import os
import nibabel as nib
from tqdm import tqdm
import json

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

if __name__ == '__main__':

    main_root = os.getcwd()
    json_file_path = os.path.join(main_root, 'sub2fmripaths.json')
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"First you need to create sub2fmripaths.json using script data_labeling.py")
    
    with open(json_file_path, 'r') as json_file:
        sub_num2paths = json.load(json_file)

    print('Creating masks for each sub ...')
    sub_num2masks = {}
    for sub, paths in tqdm(sub_num2paths.items()):
        sub_num2masks[sub] = torch.stack([get_thresholded_change_mask(torch.tensor(nib.load(path).get_fdata())) for path in paths])
    
    directory_path = 'natview/fmri_masks'
    # Create the directory and any necessary parent directories
    os.makedirs(directory_path, exist_ok=True)

    print('Combining and saving masks ...')
    for sub, masks in sub_num2masks.items():
        combined_mask = np.zeros((61, 73, 61))  # Initialize with zeros
        # Combine the masks using element-wise logical OR
        masks = masks.numpy()
        for mask in masks:
            combined_mask = np.logical_or(combined_mask, mask)
        combined_mask = torch.tensor(combined_mask)
        file_path = os.path.join(main_root, directory_path, f'{sub}.pth')
        torch.save(combined_mask, file_path)