import torch.nn as nn
import torch
import numpy as np
from eeg_fmri_fuser import EegFmriWeightFuser
from egg_encoder import EEGEncoder
from fmri_encoder import fMRIEncoder, RidgeRegression
from loss import ClipLoss

class BrainEncoder(nn.Module):
    def __init__(self, fmri_masks_path, input_length=525, num_channels=61, latent_dim=4096, embed_dim=1024):
        super().__init__()
        self.RidgeRegression = RidgeRegression(fmri_masks_path, latent_dim)
        self.fMRIEncoder = fMRIEncoder(latent_dim=latent_dim, patch_size=16, in_chans=1, embed_dim=embed_dim)
        self.EEGEncoder = EEGEncoder(input_length=input_length, num_channels=num_channels)
        self.EegFmriWeightFuser = EegFmriWeightFuser()
        self.loss_func = ClipLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, sub_ids, batch_eeg, batch_fmri):
        fmri_emb = self.RidgeRegression(sub_ids, batch_fmri)
        fmri_emb = self.fMRIEncoder(fmri_emb)
        eeg_emb = self.EEGEncoder(batch_eeg)
        fmri_eeg_emb = self.EegFmriWeightFuser(eeg_emb, fmri_emb)
        return fmri_eeg_emb
