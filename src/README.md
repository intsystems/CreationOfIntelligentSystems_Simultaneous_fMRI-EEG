# Different scripts

This repository contains source code for different parts of our model.

1. `diffusion_prior` directory contains dataset, model and pipeline for diffusion prior
2. `brain_encoder.py`, `eeg_encoder.py`, `eeg_fmri_fuser.py`, and `fmri_encoder.py` contains different modules for **BrainEncoder** model
3. `calculate_combined.py` is a script to calculate combined embeddings for the entire dataset using pretrained **BrainEncoder** model
4. `channel_recovery.py` contains code to recover missing EEG channels in our dataset
5. `dataset.py` contains dataset and dataloader for **BrainEncoder** training
6. `utils.py` contains some utilities useful for our training, e.g., `Timer` to debug different code parts

## Usage

The only script in this directory you can run is `calculate_combined.py`, and you can do it as follows:
```bash
accelerate launch calculate_combined.py
```