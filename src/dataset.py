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
import random
from channel_recovery import ChannelRecovering


def load_json_data(json_path):
    with open(json_path, "r") as file:
        data_dict = json.load(file)
    return data_dict


class BrainStimuliDataset(Dataset):
    def __init__(
        self, 
        json_path, 
        recovery_mode: str = "zeros", 
        mode: str = "both"
    ):
        self.json_path = json_path
        self.recovery_mode = recovery_mode
        self.data_dict = load_json_data(json_path)
        self.calculane_num_subs()
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
        self.drop_fmri = mode in ['none', 'eeg']
        self.drop_eeg = mode in ['none', 'fmri']
        
    def __getitem__(self, index):
        data = self.get_data_from_index(index)
        return self.process_data(
            data=data,
            drop_fmri=self.drop_fmri,
            drop_eeg=self.drop_eeg,
        )
    
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
                                'index': {'key': key, 'sub': sub, 'ses': ses, 'run': run, 'chunk': current_index},
                                'nifti_path': self.data_dict[key][sub][ses][run]['nifti_path'],
                                'eeglab_path': self.data_dict[key][sub][ses][run]['eeglab_path'],
                                'time_indices': time_indices
                            }
                        else:
                            current_index -= count
                            continue
        
    def process_data(
        self, 
        data, 
        drop_fmri: bool = False, 
        drop_eeg: bool = False
    ):
        # sub id
        _, sep, after = data['nifti_path'].partition('natview')
        id = int((sep + after).split('/')[2].split('-')[1]) - 1
        # fmri
        if not drop_fmri:
            nii_img = nib.load(data['nifti_path'])
            fmri_data = nii_img.get_fdata()
            fmri = fmri_data[:, :, :, data['time_indices']['fmri']['idx']]
            fmri = torch.from_numpy(fmri)
        else:
            fmri = None
        # eeg
        if not drop_eeg:
            raw_data = read_raw_eeglab(data['eeglab_path'])
            eeg_data = self.recover_eeg(raw_data)
            eeg = eeg_data[:, data['time_indices']['eeg']['start_idx']:data['time_indices']['eeg']['end_idx']+1]
            eeg = torch.from_numpy(eeg)
        else:
            eeg = None
        # frames
        frames_paths = [f"frame_{frame_idx:04d}.pt" for frame_idx in
                       range(data['time_indices']['frames']['start_idx'], data['time_indices']['frames']['end_idx']+1)]
        frames = list(map(lambda x: torch.load(os.path.join(data['frames_dir'], x), map_location="cuda"), frames_paths))
        frames = torch.stack(frames)
        return {
            'id': id,
            'index': data['index'],
            'fmri': fmri,
            'eeg': eeg,
            'frames': frames,
            'frame_paths': [os.path.join(data['frames_dir'], path) for path in frames_paths]
        }
        
    def calculane_num_subs(self):
        num_subs = 0
        for key in self.data_dict.keys():
            num_subs = max(num_subs, len(list(self.data_dict[key].keys())[1:]))
        self.num_subs = num_subs
        
    def calculate_length(self):
        count = 0
        for key in self.data_dict.keys():
            for sub in list(self.data_dict[key].keys())[1:]:
                for ses in self.data_dict[key][sub].keys():
                    for run in self.data_dict[key][sub][ses].keys():
                        count += len(self.data_dict[key][sub][ses][run]['chunks'])
        self.count = count
        
    def __len__(self):
        return self.count
    
    def recover_eeg(self, raw_egg):
        """Recover missing EEG channels according to mode in self.recovery_mode
        """
        # insert absent channels with nans first
        raw_egg_with_nans, nan_ids = ChannelRecovering.insert_nan_rows_in_array(raw_egg)

        # fill missing data according to chosen mode
        if self.recovery_mode == "zeros":
            return ChannelRecovering.replace_NaN_with_zeros(
                raw_egg_with_nans,
                nan_ids
            )
        elif self.recovery_mode == "kNN":
            return ChannelRecovering.replace_NaN_with_euclidean_nearest_neighbour(
                raw_egg_with_nans,
                nan_ids
            )
        elif self.recovery_mode == "kNN_weighted":
            return ChannelRecovering.replace_NaN_with_eucl_weighted_nearest_neighbour(
                raw_egg_with_nans,
                nan_ids
            )
    

def collate_fn(data):
    # get values
    id_list = [x['id'] for x in data]
    fmri_list = [x['fmri'] for x in data]
    eeg_list = [x['eeg'] for x in data]
    frames_list = [x['frames'] for x in data]
    # stack tensors
    id_tensor = torch.tensor(id_list)
    if None in fmri_list: # we drop fmri data when training only eeg encoder
        fmri_tensor = None
    else:
        fmri_tensor = torch.stack(fmri_list)
    if None in eeg_list: # we drop fmri data when training only eeg encoder
        eeg_tensor = None
    else:
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


###############################################################################


class BrainStimuliDataLoader:
    def __init__(
        self, 
        dataset, 
        batch_size, 
        mode: str = "both"
    ):
        self.batch_size = batch_size
        self.data_dict = dataset.data_dict
        self.process_data = dataset.process_data
        self.drop_fmri = mode in ['none', 'eeg']
        self.drop_eeg = mode in ['none', 'fmri']

    def __iter__(self):
        batch = []
        num_chunks_list = self.distribute_uniformly(self.batch_size, len(self.data_dict))

        for key, num_chunks in zip(self.data_dict.keys(), num_chunks_list):

            chunk_indices = []
            sub_list = list(self.data_dict[key].keys())[1:]
            sub_sample = self.custom_sample(sub_list, num_chunks)

            for sub in sub_sample:
                # sample random session
                ses_list = list(self.data_dict[key][sub].keys())
                ses = random.choice(ses_list)
                # sample random run
                run_list = list(self.data_dict[key][sub][ses].keys())
                run = random.choice(run_list)
                # sample random chunk, **that was not used before**
                chunk_list = list(self.data_dict[key][sub][ses][run]['chunks'].keys())
                # fix case when batch size > len(chunk_list)
                if len(list(set(chunk_list) - set(chunk_indices))) == 0:
                    chunk_indices = []
                chunk = random.choice(list(set(chunk_list) - set(chunk_indices)))
                chunk_indices.append(chunk)
                # append this chunk into batch
                batch.append(self.process_data(
                    data={
                        'frames_dir': self.data_dict[key]['frames_dir'],
                        'index': {'key': key, 'sub': sub, 'ses': ses, 'run': run, 'chunk': chunk},
                        'nifti_path': self.data_dict[key][sub][ses][run]['nifti_path'],
                        'eeglab_path': self.data_dict[key][sub][ses][run]['eeglab_path'],
                        'time_indices': self.data_dict[key][sub][ses][run]['chunks'][chunk]
                    },
                    drop_fmri=self.drop_fmri,
                    drop_eeg=self.drop_eeg                               
                ))
                
        yield collate_fn(batch)
        
    @staticmethod
    def distribute_uniformly(total, n):
        """
        Distribute a total number uniformly between n nodes.

        Args:
            total (int): The total number to be distributed.
            n (int): The number of nodes.

        Returns:
            list: A list of integers representing the distribution.
        """

        base = total // n
        remainder = total % n

        distribution = [base] * n

        for i in range(remainder):
            distribution[i] += 1

        return distribution
    
    @staticmethod
    def custom_sample(lst, num_samples):
        """
        Sample elements from a list in a custom manner.

        Args:
            lst (list): The list to sample from.
            num_samples (int): The number of samples to draw.

        Returns:
            list: The sampled elements.
        """
        list_len = len(lst)

        if num_samples <= list_len:
            # Sample without replacement
            return random.sample(lst, num_samples)
        else:
            # Calculate the number of full cycles and the remainder
            full_cycles = num_samples // list_len
            remainder = num_samples % list_len

            # Create the base sample with full cycles
            base_sample = lst * full_cycles

            # Sample the remainder without replacement
            remainder_sample = random.sample(lst, remainder)

            # Combine the base sample and the remainder sample
            return base_sample + remainder_sample