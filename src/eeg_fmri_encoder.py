import torch.nn as nn
from eeg_fmri_fuser import EegFmriWeightFuser
from egg_encoder import EEGEncoder
from fmri_encoder import fMRIEncoder, RidgeRegression

class BrainEncoder(nn.Module):
    def __init__(self, fmri_input_sizes, input_length=525, num_channels=61, latent_dim=4096, embed_dim=1024):
        super().__init__()
        self.RidgeRegression = RidgeRegression(input_sizes=fmri_input_sizes, out_features=latent_dim)
        self.fMRIEncoder = fMRIEncoder()
        self.EEGEncoder = EEGEncoder(input_length=input_length, num_channels=num_channels)
        self.EegFmriWeightFuser = EegFmriWeightFuser()

    def forward(self, x):
        ids, batch_eeg, batch_fmri = x
        eeg_emb = self.EEGEncoder(batch_eeg)
        fmri_emb = self.RidgeRegression(batch_fmri, ids)
        fmri_emb = self.fMRIEncoder(fmri_emb)
        fmri_eeg_emb = self.EegFmriWeightFuser(eeg_emb, fmri_emb)
        return fmri_eeg_emb
