import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from eeg_fmri_fuser import EegFmriWeightFuser, EegFmriMlpFuser
from eeg_encoder import EEGEncoder
from fmri_encoder import fMRIEncoder, RidgeRegression
from eeg_encoder import TinyEEGEncoder # try exps (new from Dorin)


class BaseBrainEncoder(nn.Module):
    """
    BaseBrainEncoder is a base class for brain encoder models.

    Attributes:
        logit_scale (nn.Parameter): A learnable parameter for scaling logits.
    """

    def __init__(self):
        super().__init__()
        # Initialize logit_scale based on the original CLIP paper
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method must be implemented by subclasses")

    def loss(self, brain_features, image_features):
        """
        Computes the loss for the brain encoder.

        Args:
            brain_features (Tensor): Brain features.
            image_features (Tensor): Image features to compare with brain features.

        Returns:
            dict: A dictionary containing the loss and logits per brain.
        """
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


class BrainEncoder(BaseBrainEncoder):
    """
    BrainEncoder is a neural network model designed to encode EEG and fMRI data
    and fuse them to produce a unified brain feature representation.

    Attributes:
        RidgeRegression (RidgeRegression): A ridge regression model for fMRI data.
        fMRIEncoder (fMRIEncoder): An encoder for fMRI data.
        EEGEncoder (EEGEncoder): An encoder for EEG data.
        EegFmriFuser (Union[EegFmriWeightFuser, EegFmriMlpFuser]): A fuser model for combining EEG and fMRI embeddings.
    """

    def __init__(self, ridge_kwargs, fmri_kwargs, eeg_kwargs, fuser_kwargs, num_subs: int = None):
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

        self.eeg_participants_embedding = nn.Embedding(num_subs, eeg_kwargs["input_length"])
        self.EEGEncoder = EEGEncoder(
            participants_embedding=self.eeg_participants_embedding,
            **eeg_kwargs
        )

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

    def forward(self, sub_ids, batch_eeg, batch_fmri, image_features: torch.Tensor = None):
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
        eeg_emb = self.EEGEncoder(batch_eeg, participant_id=sub_ids)

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

        return super().loss(brain_features, image_features)


class fMRIBrainEncoder(BaseBrainEncoder):
    """
    fMRIBrainEncoder is a neural network model designed to encode fMRI data.

    Attributes:
        RidgeRegression (RidgeRegression): A ridge regression model for fMRI data.
        fMRIEncoder (fMRIEncoder): An encoder for fMRI data.
    """

    def __init__(self, ridge_kwargs, fmri_kwargs, num_subs: int = None):
        """
        Initializes the fMRIBrainEncoder with the specified parameters.

        Args:
            ridge_kwargs (dict): Keyword arguments for RidgeRegression.
            fmri_kwargs (dict): Keyword arguments for fMRIEncoder.
        """
        super().__init__()

        self.RidgeRegression = RidgeRegression(**ridge_kwargs)
        self.fMRIEncoder = fMRIEncoder(**fmri_kwargs)

    def forward(self, sub_ids, batch_fmri, batch_eeg: torch.Tensor = None, image_features: torch.Tensor = None):
        """
        Forward pass of the fMRIBrainEncoder.

        Args:
            sub_ids (Tensor): Participant IDs.
            batch_fmri (Tensor): Batch of fMRI data.

        Returns:
            Tensor: fMRI embeddings.
        """
        # Encode fMRI data
        fmri_emb = self.RidgeRegression(batch_fmri, participant_id=sub_ids)
        fmri_emb = self.fMRIEncoder(fmri_emb)

        return fmri_emb

    def loss(self, sub_ids, batch_fmri, image_features, batch_eeg: torch.Tensor = None):
        """
        Computes the loss for the fMRIBrainEncoder.

        Args:
            sub_ids (Tensor): Participant IDs.
            batch_fmri (Tensor): Batch of fMRI data.
            image_features (Tensor): Image features to compare with brain features.

        Returns:
            dict: A dictionary containing the loss and logits per brain.
        """
        # Get brain features
        brain_features = self(
            sub_ids=sub_ids,
            batch_fmri=batch_fmri
        ).to(image_features.dtype)

        return super().loss(brain_features, image_features)


class EEGBrainEncoder(BaseBrainEncoder):
    """
    EEGBrainEncoder is a neural network model designed to encode EEG data.

    Attributes:
        EEGEncoder (EEGEncoder): An encoder for EEG data.
    """

    def __init__(self, eeg_kwargs, num_subs: int = None):
        """
        Initializes the EEGBrainEncoder with the specified parameters.

        Args:
            eeg_kwargs (dict): Keyword arguments for EEGEncoder.
        """
        super().__init__()

        # self.eeg_participants_embedding = nn.Embedding(num_subs, eeg_kwargs.input_length)
        # self.EEGEncoder = EEGEncoder(
        #     participants_embedding=self.eeg_participants_embedding,
        #     **eeg_kwargs
        # )
        self.eeg_participants_embedding = nn.Embedding(num_subs, eeg_kwargs["input_length"])
        self.EEGEncoder = TinyEEGEncoder(
            participants_embedding=self.eeg_participants_embedding,
            **eeg_kwargs
        )

    def forward(self, sub_ids, batch_eeg, batch_fmri: torch.Tensor = None, image_features: torch.Tensor = None):
        """
        Forward pass of the EEGBrainEncoder.

        Args:
            sub_ids (Tensor): Participant IDs.
            batch_eeg (Tensor): Batch of EEG data.

        Returns:
            Tensor: EEG embeddings.
        """
        # Encode EEG data
        eeg_emb = self.EEGEncoder(batch_eeg, participant_id=sub_ids)

        return eeg_emb

    def loss(self, sub_ids, batch_eeg, image_features, batch_fmri: torch.Tensor = None):
        """
        Computes the loss for the EEGBrainEncoder.

        Args:
            sub_ids (Tensor): Participant IDs.
            batch_eeg (Tensor): Batch of EEG data.
            image_features (Tensor): Image features to compare with brain features.

        Returns:
            dict: A dictionary containing the loss and logits per brain.
        """
        # Get brain features
        brain_features = self(
            sub_ids=sub_ids,
            batch_eeg=batch_eeg
        ).to(image_features.dtype)

        return super().loss(brain_features, image_features)
