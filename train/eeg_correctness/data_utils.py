from pathlib import Path
import re

import numpy as np
import torch

from torch.utils.data import Dataset, Sampler, DataLoader, DistributedSampler


class EegImgLatentDataset(Dataset):
    def __init__(
        self,
        img_latent_path: Path,
        eeg_paths: dict[int, Path],    # map of paths participant_id -> participant file
    ):
        """ each eeg has dim=(img_num, trial_num, channel, time)
        """
        super().__init__()

        # img latents can be loaded fully
        self._img_latents: torch.Tensor = torch.load(img_latent_path, weights_only=True)["img_features"]

        # load eeg's using mmap (not fully in RAM)
        self._eegs = [
            np.load(eeg_path, mmap_mode='r') for eeg_path in eeg_paths.values()
        ]
        # save participants' ids
        self._participants = list(eeg_paths.keys())

        # it is assumed that each participant has equal number of eegs
        # it equals num_imgs * num_trials
        self._eeg_set_shape = self._eegs[0].shape
        self._eegs_per_partic = self._eeg_set_shape[0] * self._eeg_set_shape[1]

        # compute total number of eegs
        self._total_eegs = sum(list(eeg.shape[0] * eeg.shape[1] for eeg in self._eegs))

    def __len__(self):
        return self._total_eegs

    def __getitem__(self, indx):
        """ returns tuple(participant_id, eeg, img_latent)
        """
        partic_num = indx // self._eegs_per_partic
        eeg_num = indx % self._eegs_per_partic
        img_num = eeg_num // self._eeg_set_shape[1]
        trial_num = eeg_num % self._eeg_set_shape[1]

        return (
            self._participants[partic_num],
            torch.from_numpy(self._eegs[partic_num][img_num, trial_num]).float(),
            self._img_latents[img_num]
        )


class ClipSampler(Sampler):
    """ Orders eeg examples so it is basically strips of unique images.
        Is used to get diverse batches for CLIP-loss training.
    """
    def __init__(self, dataset: EegImgLatentDataset):
        """_summary_

        Args:
            dataset (EegImgLatentDataset): is used as a friend class
        """
        self._num_eegs = len(dataset._eegs)
        self._num_trials = dataset._eeg_set_shape[1]
        self._num_imgs = dataset._eeg_set_shape[0]

    def _get_random_grouped_indxs(self):
        rand_gen = np.random.default_rng()

        all_indxs = np.arange(0, self._num_eegs * self._num_trials * self._num_imgs)\
                      .reshape(self._num_eegs, self._num_imgs, self._num_trials)\
                      .swapaxes(0, 1)\
                      .reshape(-1, self._num_imgs)
        # permute img examples for all images
        all_indxs = rand_gen.permuted(all_indxs, axis=0, out=all_indxs)
        # permute order of the img examples in each row
        all_indxs = rand_gen.permuted(all_indxs, axis=1, out=all_indxs)

        return all_indxs

    def __iter__(self):
        all_indxs = self._get_random_grouped_indxs()

        return all_indxs.flat

    def __len__(self):
        return self._num_eegs * self._num_trials * self._num_imgs


class DistributedClipSampler(DistributedSampler, ClipSampler):
    """ distributed version of the ClipSampler
    """
    def __init__(self, dataset, num_replicas = None, rank = None, shuffle = True, seed = 0, drop_last = False):
        DistributedSampler.__init__(self, dataset, num_replicas, rank, shuffle, seed, drop_last)
        ClipSampler.__init__(self, dataset)

    def __len__(self):
        return ClipSampler.__len__(self)

    def __iter__(self):
        all_indxs = self._get_random_grouped_indxs()

        chunk_start = (len(self) // self.num_replicas) * self.num_replicas
        chunk_end = min(
            (len(self) // self.num_replicas) * (self.num_replicas + 1),
            len(self)
        )

        # flatten indicies array and return corresponding chunk
        return np.ravel(all_indxs)[chunk_start:chunk_end].flat


def eeg_img_latent_collate(batch_list: list) -> torch.Tensor:
    participants = torch.LongTensor([exmpl[0] for exmpl in batch_list])
    eegs = torch.stack([exmpl[1] for exmpl in batch_list])
    img_latents = torch.stack([exmpl[2] for exmpl in batch_list])

    return participants, eegs, img_latents


def parse_subs_eeg_dir(subs_dir: Path):
    subs = {}

    for sub_dir in subs_dir.glob("sub-*"):
        sub_num = int(
            re.match(r"sub-(?P<sub_num>\d*)", sub_dir.stem).groupdict()["sub_num"]
        )
        train_file = next(sub_dir.glob("*train*"))
        test_file = next(sub_dir.glob("*test*"))
        
        subs[sub_num] = {"train": train_file, "test": test_file}

    return subs
