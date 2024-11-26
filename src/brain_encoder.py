import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from eeg_fmri_fuser import EegFmriWeightFuser, EegFmriMlpFuser
from eeg_encoder import EEGEncoder
from fmri_encoder import fMRIEncoder, RidgeRegression

class BrainEncoder(nn.Module):
    """
    BrainEncoder is a neural network model designed to encode EEG and fMRI data
    and fuse them to produce a unified brain feature representation.

    Attributes:
        RidgeRegression (RidgeRegression): A ridge regression model for fMRI data.
        fMRIEncoder (fMRIEncoder): An encoder for fMRI data.
        EEGEncoder (EEGEncoder): An encoder for EEG data.
        EegFmriFuser (Union[EegFmriWeightFuser, EegFmriMlpFuser]): A fuser model for combining EEG and fMRI embeddings.
        logit_scale (nn.Parameter): A learnable parameter for scaling logits.
    """

    def __init__(self, ridge_kwargs, fmri_kwargs, eeg_kwargs, fuser_kwargs):
        """
        Initializes the BrainEncoder with the specified parameters.

        Args:
            ridge_kwargs (dict): Keyword arguments for RidgeRegression.
            fmri_kwargs (dict): Keyword arguments for fMRIEncoder.
            eeg_kwargs (dict): Keyword arguments for EEGEncoder.
            fuser_kwargs (dict): Keyword arguments for the fuser model.
        """
        super().__init__()
        self.RidgeRegression = RidgeRegression(**ridge_kwargs)
        self.fMRIEncoder = fMRIEncoder(**fmri_kwargs)
        self.EEGEncoder = EEGEncoder(**eeg_kwargs)

        # Determine the type of fuser to use based on the provided fuser_kwargs
        fuser_name = fuser_kwargs.get('fuser_name')
        if fuser_name == 'weight':
            self.EegFmriFuser = EegFmriWeightFuser()
        elif fuser_name == 'mlp':
            emb_dim = fuser_kwargs.get('emb_dim')
            if emb_dim is None:
                raise ValueError("emb_dim must be specified for MLP fuser")
            self.EegFmriFuser = EegFmriMlpFuser(emb_dim=emb_dim)
        else:
            raise NotImplementedError(f"Fuser type {fuser_name} is not implemented")

        # Initialize logit_scale based on the original CLIP paper
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, sub_ids, batch_eeg, batch_fmri):
        """
        Forward pass of the BrainEncoder.

        Args:
            sub_ids (Tensor): Participant IDs.
            batch_eeg (Tensor): Batch of EEG data.
            batch_fmri (Tensor): Batch of fMRI data.

        Returns:
            Tensor: Fused EEG and fMRI embeddings.
        """
        # Encode fMRI data
        fmri_emb = self.RidgeRegression(batch_fmri, participant_id=sub_ids)
        fmri_emb = self.fMRIEncoder(fmri_emb)

        # Encode EEG data
        eeg_emb = self.EEGEncoder(batch_eeg)
        # eeg_emb = self.EEGEncoder(batch_eeg, participant_id=sub_ids)

        # Fuse EEG and fMRI embeddings
        fmri_eeg_emb = self.EegFmriFuser(eeg_emb=eeg_emb, fmri_emb=fmri_emb)

        return fmri_eeg_emb

    def loss(self, sub_ids, batch_eeg, batch_fmri, image_features):
        """
        Computes the loss for the BrainEncoder.

        Args:
            sub_ids (Tensor): Participant IDs.
            batch_eeg (Tensor): Batch of EEG data.
            batch_fmri (Tensor): Batch of fMRI data.
            image_features (Tensor): Image features to compare with brain features.

        Returns:
            dict: A dictionary containing the loss and logits per brain.
        """
        # Get brain features
        brain_features = self(
            sub_ids=sub_ids,
            batch_eeg=batch_eeg,
            batch_fmri=batch_fmri
        ).to(image_features.dtype)

        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        brain_features = brain_features / brain_features.norm(dim=1, keepdim=True)

        # Compute cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ brain_features.T
        logits_per_brain = logits_per_image.T

        # Compute cross-entropy loss
        batch_size = logits_per_image.shape[0]
        labels = torch.arange(batch_size, device=image_features.device, dtype=torch.long)

        total_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_brain, labels)) / 2

        return {'loss': total_loss, 'logits_per_brain': logits_per_brain}

    @property
    def device(self):
        """
        Returns the device of the model parameters.

        Returns:
            torch.device: The device of the model parameters.
        """
        return next(iter(self.parameters()))[0].device
