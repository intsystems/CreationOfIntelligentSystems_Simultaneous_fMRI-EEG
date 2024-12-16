import nibabel as nib
from mne.io import read_raw_eeglab
import os
import nibabel as nib
from mne.io import read_raw_eeglab
import torch

from torch.utils.data import Dataset, DataLoader, random_split
import dataset


class BrainStimuliDatasetEEG(dataset.BrainStimuliDataset):
    """ Copy of dataset.BrainStimuliDataset, but do not load fmri from disk
    """
    def __init__(self, json_path, recovery_mode = "zeros"):
        super().__init__(json_path, recovery_mode)

    def process_data(self, data):
        # sub id
        _, sep, after = data['nifti_path'].partition('natview')
        id = int((sep + after).split('/')[2].split('-')[1]) - 1
        # no fmri
        # 
        # eeg
        raw_data = read_raw_eeglab(data['eeglab_path'])
        eeg_data = self.recover_eeg(raw_data)
        eeg = eeg_data[:, data['time_indices']['eeg']['start_idx']:data['time_indices']['eeg']['end_idx']+1]
        eeg = torch.from_numpy(eeg)
        # frames
        frames_paths = [f"frame_{frame_idx:04d}.pt" for frame_idx in
                       range(data['time_indices']['frames']['start_idx'], data['time_indices']['frames']['end_idx']+1)]
        frames = list(map(lambda x: torch.load(os.path.join(data['frames_dir'], x), map_location="cuda"), frames_paths))
        frames = torch.stack(frames)
        return {
            'id': id,
            'index': data['index'],
            'eeg': eeg,
            'frames': frames,
            'frame_paths': [os.path.join(data['frames_dir'], path) for path in frames_paths]
        }


def collate_fn(data):
    """Copy of dataset.collate_fn but do not use fmri
    """
    # get values
    id_list = [x['id'] for x in data]
    eeg_list = [x['eeg'] for x in data]
    frames_list = [x['frames'] for x in data]
    # stack tensors
    id_tensor = torch.tensor(id_list)
    eeg_tensor = torch.stack(eeg_list)
    frames_tensor = torch.stack(frames_list)
    return {
        'id': id_tensor,
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
    dataset = BrainStimuliDatasetEEG(dataset_json)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, val_dataloader