from collections import OrderedDict
from copy import copy

import numpy as np
import mne

class ChannelRecovering:
    '''
        all methods, used to recover missing eeg channels, based on existing
        important: part of code was apatped from https://github.com/intsystems/CreationOfIntelligentSystems_Simultaneous_fMRI-EEG/blob/data/code/data/dataset.ipynb

        ALL methods are static

        You need methods:

    '''

    all_eeg_channels_ordered = [
            'AF3', 'AF4', 'AF7', 'AF8', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
            'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz', 'Cz', 'F1', 'F2',
            'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4',
            'FC5', 'FC6', 'FT7', 'FT8', 'Fp1', 'Fp2', 'Fpz', 'Fz', 'O1', 'O2',
            'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO4',
            'PO7', 'PO8', 'POz', 'Pz', 'T7', 'T8', 'TP10', 'TP7', 'TP8', 'TP9'
    ]

    EPS = 1e-3

    def __init__(self):
        pass

    ### common method for inserting NaNs for unpresen channels
    @staticmethod
    def insert_nan_rows_in_array(raw_data):
        '''
            adds novel channels into raw_data and fiils it with Nones

            returns:
                - raw_data (np.array) with Nans of shape [n_channels, time]
                - nan_ids (list[int]) with indicies of nan channels (they will be recovered later)

            important: code was apatped from https://github.com/intsystems/CreationOfIntelligentSystems_Simultaneous_fMRI-EEG/blob/data/code/data/dataset.ipynb
        '''

        all_ordered_channels = ChannelRecovering.all_eeg_channels_ordered

        raw_data_ordered = raw_data.reorder_channels(sorted(list(set(all_ordered_channels) & set(raw_data.ch_names))))
        current_channels_ordered = raw_data_ordered.ch_names
        good_indices = []
        i = 0
        j = 0
        while i < len(current_channels_ordered) and j < len(all_ordered_channels):
            if current_channels_ordered[i] == all_ordered_channels[j]:
                good_indices.append(j)
                i += 1
                j += 1
            else:
                j += 1
        raw_data_array = raw_data_ordered.get_data()
        raw_data_array_with_inserted_NaN_rows = np.empty(shape=(len(all_ordered_channels), raw_data_array.shape[1])) # np.zeros((len(all_eeg_channels_ordered), raw_data_array.shape[1]))
        raw_data_array_with_inserted_NaN_rows[:] = np.nan

        for i, idx in enumerate(good_indices):
            raw_data_array_with_inserted_NaN_rows[idx] = raw_data_array[i]


        NaN_ids = np.arange(len(all_ordered_channels))[np.isnan(raw_data_array_with_inserted_NaN_rows).all(axis=1)]

        return raw_data_array_with_inserted_NaN_rows, NaN_ids

    ### 2 functions below for eucludean NN replace
    @staticmethod
    def _find_existing_euclidean_nn(nan_ids, curr_id, all_values):
        '''
            all_values: np.array [n_channels, 3] with spaial coordinates

            find nearest to curr_id channel, that is not contained in nan_ids and != curr_id
        '''

        sq_dists = ((all_values - all_values[curr_id])**2).mean(axis=1) # [n_channels]

        sorted_nearest_ids = np.argsort(sq_dists)

        for near_id in sorted_nearest_ids:
            if near_id not in nan_ids and near_id != curr_id:
                return near_id

        assert False, f"Failed to found nearest existing to {curr_id}. nan_ids: {nan_ids.shape}"

    @staticmethod
    def replace_NaN_with_euclidean_nearest_neighbour(eeg_with_nans, nan_ids):
        eeg_with_nans = eeg_with_nans.copy()
        # Load a standard montage
        montage = mne.channels.make_standard_montage('standard_1020')

        # Get the positions of your channels
        channel_positions = montage.get_positions()['ch_pos']

        ordered = OrderedDict((k, channel_positions[k]) for k in ChannelRecovering.all_eeg_channels_ordered)
        all_values = np.array([ordered[ch_name] for ch_name in ChannelRecovering.all_eeg_channels_ordered])

        replace_ids = []
        for nan_id in nan_ids:
            rep_id = ChannelRecovering._find_existing_euclidean_nn(nan_ids, nan_id, all_values)
            replace_ids.append(rep_id)

        for nan_id, rep_id in zip(nan_ids, replace_ids):
            eeg_with_nans[nan_id] = eeg_with_nans[rep_id]

        return eeg_with_nans

    ### baseline: replace with zeros
    @staticmethod
    def replace_NaN_with_zeros(eeg_with_nans, nan_ids):
        eeg_with_nans = eeg_with_nans.copy()


        # Load a standard montage
        montage = mne.channels.make_standard_montage('standard_1020')

        # Get the positions of your channels
        channel_positions = montage.get_positions()['ch_pos']

        ordered = OrderedDict((k, channel_positions[k]) for k in ChannelRecovering.all_eeg_channels_ordered)
        all_values = np.array([ordered[ch_name] for ch_name in ChannelRecovering.all_eeg_channels_ordered])

        for nan_id in nan_ids:
            eeg_with_nans[nan_id] = 0

        return eeg_with_nans



    ### 2 functions below for advanced kNN replacement
    @staticmethod
    def _find_k_NNs_with_dist_weights_except(k, curr_id, sensors_coordinate, except_lst):

        assert k > 0

        dists = np.sqrt(((sensors_coordinate - sensors_coordinate[curr_id])**2).mean(axis=1)) # [n_channels]

        sorted_nearest_ids = np.argsort(dists)

        found_idx = []
        found_dists = []

        for near_id in sorted_nearest_ids:
            if near_id not in except_lst:
                found_idx.append(near_id)
                found_dists.append(dists[near_id])

                if len(found_idx) == k:
                    return found_idx, found_dists


        assert False, f"Failed to found nearest existing to {curr_id}."


    @staticmethod
    def replace_NaN_with_eucl_weighted_nearest_neighbour(eeg_with_nans, nan_ids, n_neighbours=3):
        eeg_with_nans = eeg_with_nans.copy()
        # Load a standard montage
        montage = mne.channels.make_standard_montage('standard_1020')

        # Get the positions of your channels
        channel_positions = montage.get_positions()['ch_pos']

        ordered = OrderedDict((k, channel_positions[k]) for k in ChannelRecovering.all_eeg_channels_ordered)
        all_values = np.array([ordered[ch_name] for ch_name in ChannelRecovering.all_eeg_channels_ordered])

        replace_into = []
        for nan_id in nan_ids:
            rep_ids, rep_dists = ChannelRecovering._find_k_NNs_with_dist_weights_except(n_neighbours, nan_id, all_values, nan_ids)

            weights = 1 / (np.array(rep_dists) + ChannelRecovering.EPS)[:, None]

            replace_into.append((eeg_with_nans[rep_ids] * weights).sum(axis=0) / weights.sum())

        for nan_id, rep_into in zip(nan_ids, replace_into):
            eeg_with_nans[nan_id] = rep_into

        return eeg_with_nans