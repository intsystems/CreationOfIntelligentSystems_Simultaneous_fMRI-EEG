import nibabel as nib
from mne.io import read_raw_eeglab
import os
import pandas as pd
import glob
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings("ignore")

###############################################################################

SKIPPED_FMRI_IMAGES = 2 # we will skip first SKIPPED_FMRI_IMAGES to pay attention to BOLD delay
FPS = 3 # video frequency
FMRI_TR = 2.1 # temporal resolution of fMRI, i.e. the frequency
EEG_TR = 0.004 # the same as for fMRI, but for EEG

###############################################################################

class AutoVivification(dict):
    """
    Implementation of perl's autovivification feature.
    This is very convenient way to use nested dicts, adding new elements.
    """
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value
        
def chunk_idx_to_fmri_index(chunk_idx, num_skipped_fmri_images=2):
    return chunk_idx + num_skipped_fmri_images

def chunk_idx_to_eeg_time_indices(chunk_idx, fmri_tr=2.1, eeg_tr=0.004):
    eeg_start_idx = int(chunk_idx * fmri_tr / eeg_tr)
    eeg_end_idx = int(eeg_start_idx + fmri_tr / eeg_tr) - 1
    return {'start_idx': eeg_start_idx, 'end_idx': eeg_end_idx}

def chunk_idx_to_frames_time_indices(chunk_idx, fmri_tr=2.1, video_fps=3):
    frames_start_idx = int(chunk_idx * fmri_tr * video_fps)
    frames_end_idx = int(frames_start_idx + fmri_tr * video_fps) - 1
    return {'start_idx': frames_start_idx, 'end_idx': frames_end_idx}

###############################################################################

if __name__ == '__main__':

    root = './natview/data'
    stimuli_dir = os.path.join(root, 'stimuli')
    filtered_paths = glob.glob(os.path.join(root, '*/*/func/*/func_preproc/func_pp_filter_gsr_sm0.mni152.3mm.nii.gz'))
    # filtered_paths = glob.glob(os.path.join(root, '*/*/func/*/func_preproc/func_pp_filter_sm0.mni152.3mm.nii.gz'))

    videoname2key = {
        'Despicable Me (English)': 'dme',
        'Despicable Me (Hungarian)': 'dmh',
        'The Present': 'tp',
        'Inscapes': 'inscapes',
        'Monkey 1': 'monkey1',
        'Monkey 2': 'monkey2',
        'Monkey 5': 'monkey5',
    }

    key2videoname = {key: videoname for videoname, key in videoname2key.items()}
    key2paths = {}
    for key in key2videoname.keys():
        key2paths[key] = sorted(list(filter(lambda x: key in x, filtered_paths)))
        
    # key -> 1) frames_dir, 2) sub
    # sub -> ses -> run
    # run -> 1) nifti_path, 2) eeglab_path, 3) chunks
    # chunk -> 1) fmri, 2) eeg, 3) frames
    # eeg -> 1) start_idx, 2) end_idx
    # frames -> 1) start_idx, 2) end_idx
    
    data_dict = AutoVivification()

    for key in tqdm(key2paths.keys()):
        # frames directory
        frames_dir = os.path.join(stimuli_dir, key)
        data_dict[key]['frames_dir'] = frames_dir
        # loop for path to the fmri image
        for nifti_path in key2paths[key]:
            eeglab_path = os.path.join(*nifti_path.split('/')[:-2]).replace('func', 'eeg').replace('bold', 'eeg') + '.set'
            divided_str = nifti_path[nifti_path.rfind('func/') + len('func/'):].split('/')[0].split('_')
            if len(divided_str) == 4:
                sub_str, ses_str = divided_str[:2]
                run_str = 'run-01'
            elif len(divided_str) == 5:
                sub_str, ses_str, _, run_str, _ = divided_str
            else:
                raise ValueError
            # calculate number of chunks
            nii_img = nib.load(nifti_path)
            try:
                raw_data = read_raw_eeglab(eeglab_path)
            except FileNotFoundError:
                print(f"EEG data for {sub_str}/{ses_str}/{run_str} not found.")
                continue
            num_fmri_images = nii_img.shape[-1] - SKIPPED_FMRI_IMAGES
            num_eeg_chunks = int(raw_data.n_times // (FMRI_TR / EEG_TR))
            num_frames_chunks = int(len(os.listdir(frames_dir)) // (FMRI_TR * FPS))
            num_chunks = min(num_fmri_images, num_eeg_chunks, num_frames_chunks)
            # print('num_fmri_images:', num_fmri_images)
            # print('num_eeg_chunks:', num_eeg_chunks)
            # print('num_frames_chunks:', num_frames_chunks)
            # print('num_chunks:', num_chunks)
            # fmri
            data_dict[key][sub_str][ses_str][run_str]['nifti_path'] = nifti_path
            # eeg
            data_dict[key][sub_str][ses_str][run_str]['eeglab_path'] = eeglab_path
            for chunk_idx in range(num_chunks):
                # fmri
                data_dict[key][sub_str][ses_str][run_str]['chunks'][chunk_idx]['fmri']['idx'] = chunk_idx_to_fmri_index(chunk_idx, SKIPPED_FMRI_IMAGES)
                # eeg
                data_dict[key][sub_str][ses_str][run_str]['chunks'][chunk_idx]['eeg'] = chunk_idx_to_eeg_time_indices(chunk_idx, FMRI_TR, EEG_TR)
                # frames
                data_dict[key][sub_str][ses_str][run_str]['chunks'][chunk_idx]['frames'] = chunk_idx_to_frames_time_indices(chunk_idx, FMRI_TR, FPS)

    with open("dataset.json", "w") as outfile:
        json.dump(data_dict, outfile, indent=4)