# Data

This repository contains the files to
1. Load the data from the web source
2. Visualize the fMRI and EEG samples
3. Create the entire `dataset.json`, containing the data paths
4. Initialize `Dataset`, `DataLoader` and iterate over the latter one

> [!NOTE]
> In order to run the project successfully, please follow the next instructions.
> You should create new conda environment and run the scripts in the particular order.

This document is structured as follows:

1. [Installation](#installation)
2. [Usage](#usage)

## Installation <a name="installation"></a>

To run all the scripts, you need to have Python and the required packages installed on your computer.

Clone the repository:
```bash
git clone https://github.com/intsystems/CreationOfIntelligentSystems_Simultaneous_fMRI-EEG.git
```

Navigate to the repository directory:
```bash
cd CreationOfIntelligentSystems_Simultaneous_fMRI-EEG/code
```

Create a conda environment using the provided `environment.yml` file:
```bash
conda env create -f environment.yml
```

Activate the conda environment:
```bash
conda activate visual_stimuli
```

## Usage <a name="usage"></a>

1. Firstly, load the data from the web sources:
    ```bash
    bash load_natview_data.sh
    ```
    This script will download necessary data files: the dataset with simultaneous fMRI-EEG and visual stimuli that include multiple videos.
    The directory `.natview/data` will be created, in which the files for each `sub` will be downloaded, as the `stimuli` directory, containing the videos in `.avi` format.
2. If you wish, you can run the cells in the `data_visualization.ipynb` notebook. There you can see the fMRI and EEG samples from the downloaded dataset.
3. Further, run the `data_labeling.py`. It creates the `dataset.json` file containing the data paths. Here we provide an example of its structure. First key `dme` refers to the video stimuli name key (`dme` means `Despicable me, English version`). For each video stimuli we have created its own directory (see `load_natview_data.sh` source file), which correponds to the `frames_dir` key. In the dataset, there is data for many participants named `sub-01`, `sub-02`, ..., `sub-22`, connected with the same key name. As well as multiple session, there are multiple runs for each participants, e.g. keys `ses-01` and `run-01`. Each particular run has its own fMRI and EEG data that are saved in the `nifti_path` and `eeglab_path`. As we work with "chunks" of data, i.e. certain triplets (fMRI, EEG, frames), we store the data in the same format. Thus, for each run we have many chunks, number of which depends on the particular run, but is often around ~200. To connect fMRI, EEG and frames data with each other we store the corresponding time indices. As we map only one fMRI image to multiple EEG and frames, `fmri` has only one key `idx`, while `eeg` and `frames` have two indices: `start_idx` and `end_idx`. These indices refer to the particular time points in the corresponding time series.
    ```json
    {
    "dme": {
        "frames_dir": "path/to/video/stimuli/dir",
        "sub-01": {
            "ses-01": {
                "run-01": {
                    "nifti_path": "path/to/file.nii.gz",
                    "eeglab_path": "path/to/file.set",
                    "chunks": {
                        "0": {"fmri": {"idx": 2}, "eeg": {"start_idx": 0, "end_idx": 524}, "frames": {"start_idx": 0, "end_idx": 5}},
                        "1": {"fmri": { "idx": 3}, "eeg": {"start_idx": 525, "end_idx": 1049}, "frames": { "start_idx": 6, "end_idx": 11}},
                    }
                },
            },
        },
    },
    }
    ```
4. When the `dataset.json` file was created, you can initialize the custom `Dataset`, inherited from the `torch.utils.data.Dataset`, and then loop over the corresponding `DataLoader`. To do it, run the cells in the `dataset.ipynb` notebook.