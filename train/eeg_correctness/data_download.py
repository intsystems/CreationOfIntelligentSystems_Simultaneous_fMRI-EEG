""" Script for downloading EEG and image latentns from huggingface hub repo "LidongYang/EEG_Image_decode"
"""
from pathlib import Path
import re

import numpy as np

from huggingface_hub import snapshot_download


if __name__ == "__main__":
    # download first 5 subjects from THINGS_EEG
    # and image latents
    snapshot_download(
        repo_id="LidongYang/EEG_Image_decode",
        repo_type="dataset",
        allow_patterns=[
            "Preprocessed_data_250Hz/sub-0[1-5]/*",
            "ViT-H-14_features_test.pt",
            "ViT-H-14_features_train.pt"
        ],
        local_dir="data"
    )

    # make all subjects ids be in range(0, N)
    for sub_dir in Path("data/Preprocessed_data_250Hz").glob("*"):
        print(f"Rewriting train eeg for {sub_dir}...")
        # remove all redundant info, leave only EEG
        train = np.load(sub_dir / "preprocessed_eeg_training.npy", allow_pickle=True)["preprocessed_eeg_data"]
        np.save(sub_dir / "preprocessed_eeg_training.npy", train, allow_pickle=False)
        del(train)
        
        print(f"Rewriting test eeg for {sub_dir}...")
        test = np.load(sub_dir / "preprocessed_eeg_test.npy", allow_pickle=True)["preprocessed_eeg_data"]
        np.save(sub_dir / "preprocessed_eeg_test.npy", test, allow_pickle=False)
        del(test)

        # decrease sub's number and rename dir
        sub_num = int(
            re.match(r"sub-(?P<sub_num>\d*)", sub_dir.stem).groupdict()["sub_num"]
        )
        sub_dir.rename(sub_dir.parent / f"sub-{sub_num - 1}")
