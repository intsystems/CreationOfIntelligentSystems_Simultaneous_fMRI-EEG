import nibabel as nib
from mne.io import read_raw_eeglab
import os
import pandas as pd
import glob
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib
from mne.io import read_raw_eeglab
import torch
from PIL import Image
from torchvision import transforms
import json


class BrainStimuliDataset(Dataset):
    def __init__(self, json_path, frame_size=224):
        self.json_path = json_path
        self.frame_size = frame_size
        self.data_dict = self.load_json_data(json_path)
        self.calculate_length()
        # union of all the available channels in EEG experiments
        self.eeg_channels_ordered = [
            'AF3', 'AF4', 'AF7', 'AF8', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 
            'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz', 'Cz', 'F1', 'F2',
            'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4',
            'FC5', 'FC6', 'FT7', 'FT8', 'Fp1', 'Fp2', 'Fpz', 'Fz', 'O1', 'O2',
            'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO4',
            'PO7', 'PO8', 'POz', 'Pz', 'T7', 'T8', 'TP10', 'TP7', 'TP8', 'TP9'
        ]
        
    def __getitem__(self, index):
        if isinstance(index, int):
            index = [index]
        elif isinstance(index, slice):
            index = list(range(index.start or 0, index.stop or len(self), index.step or 1))
        else:
            raise TypeError('{cls} indices must be integers or slices, not {idx}'.format(
                cls=type(self).__name__,
                idx=type(index).__name__,
            ))
        data_list = [self.get_data_from_index(idx) for idx in index]
        # initialize lists
        fmri_list = []
        eeg_list = []
        frames_list = []
        id_list = []
        for data in data_list:
            # sub id
            _, sep, after = data['nifti_path'].partition('natview')
            id = int((sep + after).split('/')[3].split('-')[1])
            id_list.append(id)
            # fmri
            nii_img = nib.load(data['nifti_path'])
            fmri_data = nii_img.get_fdata()
            fmri = fmri_data[:, :, :, data['time_indices']['fmri']['idx']]
            fmri_list.append(torch.from_numpy(fmri).to(dtype=torch.float))
            # eeg
            raw_data = read_raw_eeglab(data['eeglab_path'])
            eeg_data = self.insert_zero_rows_in_array(raw_data)
            eeg = eeg_data[:, data['time_indices']['eeg']['start_idx']:data['time_indices']['eeg']['end_idx']+1]
            eeg_list.append(torch.from_numpy(eeg).to(dtype=torch.float))
            # frames
            frames_paths = [f"frame_{frame_idx:04d}.pt" for frame_idx in 
                           range(data['time_indices']['frames']['start_idx'], data['time_indices']['frames']['end_idx']+1)]
            frames = list(map(lambda x: torch.load(os.path.join(data['frames_dir'], x), map_location="cuda"), frames_paths))
            frames_list.append(torch.stack(frames))
        # stack tensors
        id_tensor = torch.tensor(id_list)
        fmri_tensor = torch.stack(fmri_list)
        eeg_tensor = torch.stack(eeg_list)
        frames_tensor = torch.stack(frames_list)
        # remove first dim for integer index 
        if len(index) == 1:
            fmri_tensor = fmri_tensor.squeeze(0)
            eeg_tensor = eeg_tensor.squeeze(0)
            frames_tensor = frames_tensor.squeeze(0)
        return {
            'id': id_tensor,
            'fmri': fmri_tensor,
            'eeg': eeg_tensor,
            'frames': frames_tensor
        }
        
    def __len__(self):
        return self.count
    
    def load_json_data(self, json_path):
        with open(json_path, "r") as file:
            data_dict = json.load(file)
        return data_dict
    
    def calculate_length(self):
        count = 0
        for key in self.data_dict.keys():
            for sub in list(self.data_dict[key].keys())[1:]:
                for ses in self.data_dict[key][sub].keys():
                    for run in self.data_dict[key][sub][ses].keys():
                        count += len(self.data_dict[key][sub][ses][run]['chunks'])
        self.count = count
        
    def get_data_from_index(self, idx):
        """Should be enhanced for multiple indices, as it is called during the `self.__getitem__()`"""
        current_index = idx    
        for key in self.data_dict.keys():
            for sub in list(self.data_dict[key].keys())[1:]:
                for ses in self.data_dict[key][sub].keys():
                    for run in self.data_dict[key][sub][ses].keys():
                        count = len(self.data_dict[key][sub][ses][run]['chunks'])
                        if current_index < count:
                            time_indices = self.data_dict[key][sub][ses][run]['chunks'][str(current_index)]
                            return {
                                'frames_dir': self.data_dict[key]['frames_dir'],
                                'nifti_path': self.data_dict[key][sub][ses][run]['nifti_path'],
                                'eeglab_path': self.data_dict[key][sub][ses][run]['eeglab_path'],
                                'time_indices': time_indices
                            }
                        else:
                            current_index -= count
                            continue
                        
    def insert_zero_rows_in_array(self, raw_data):
        """Sort channels in `raw_data`, and then insert zero rows for channels that are not included in `raw_data.ch_names`"""
        raw_data_ordered = raw_data.reorder_channels(sorted(list(set(self.eeg_channels_ordered) & set(raw_data.ch_names))))
        current_channels_ordered = raw_data_ordered.ch_names
        good_indices = []
        i = 0
        j = 0
        while i < len(current_channels_ordered) and j < len(self.eeg_channels_ordered):
            if current_channels_ordered[i] == self.eeg_channels_ordered[j]:
                good_indices.append(j)    
                i += 1
                j += 1
            else:
                j += 1
        raw_data_array = raw_data_ordered.get_data()
        raw_data_array_with_inserted_zero_rows = np.zeros((len(self.eeg_channels_ordered), raw_data_array.shape[1]))
        for i, idx in enumerate(good_indices):
            raw_data_array_with_inserted_zero_rows[idx] = raw_data_array[i]
        return raw_data_array_with_inserted_zero_rows
    

def collate_fn(data):
    # get values
    id_list = [x['id'] for x in data]
    fmri_list = [x['fmri'] for x in data]
    eeg_list = [x['eeg'] for x in data]
    frames_list = [x['frames'] for x in data]
    # stack tensors
    id_tensor = torch.tensor(id_list)
    fmri_tensor = torch.stack(fmri_list)
    eeg_tensor = torch.stack(eeg_list)
    frames_tensor = torch.stack(frames_list)
    return {
        'id': id_tensor,
        'fmri': fmri_tensor,
        'eeg': eeg_tensor, 
        'frames': frames_tensor
    }

def build_dataloaders(dataset_json: str, batch_size: int, train_ratio: float = 0.9) -> tuple[DataLoader]:
    """ Builds train/validate dataloaders for (id, eeg, fmri, imgs) triplets

    Args:
        train_ratio (float): data ratio for train, should be in (0., 1.)

    Returns:
        tuple[DataLoader]: train/validate dataloaders
    """
    dataset = BrainStimuliDataset(dataset_json)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, val_dataloader


def select_random_dimension(batch):
    """
    Selects a random dimension for each object in the batch and returns a batch of size (batch_size, 1024).

    :param batch: Tensor of shape (batch_size, num_dimensions, 1024)
    :return: Tensor of shape (batch_size, 1024)
    """
    batch_size, num_dimensions, _ = batch.size()
    # Generate random indices for each object in the batch
    random_indices = torch.randint(0, num_dimensions, (batch_size,))
    # Use the random indices to select the corresponding dimensions
    return batch[torch.arange(batch_size), random_indices]