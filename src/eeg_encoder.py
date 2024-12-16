from typing import Union, Callable, Optional
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class SpatioChannelConv(nn.Module):
    """ Module takes channeled signals (num_channel, time) and make 
         separate full time and full channel convolution. The output is vector.
    """
    def __init__(
            self,
            input_length: int,
            num_channels: int,
            output_dim: int,
            kernal_size: int
    ):
        super().__init__()

        # compute the stride for time convolutions so the time dim will be almost wrapped after
        stride = self._compute_stride(input_length, kernal_size)

        self.time_conv = nn.Sequential(
            nn.Conv2d(1, output_dim // 2, (1 ,kernal_size), (1, stride)),
            nn.AvgPool2d((1 ,kernal_size), (1, stride))
        )

        self.channel_conv = nn.Sequential(
            nn.BatchNorm2d(output_dim // 2),
            nn.ELU(),
            nn.Conv2d(output_dim // 2, output_dim, (num_channels, 1)),
            nn.ELU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add feature maps dimension
        x.unsqueeze_(dim=1)
        # apply time convolutions
        x = self.time_conv(x)
        # average the remained time dimension
        x = torch.mean(x, dim=3, keepdim=True)
        # apply channel convolutions
        x = self.channel_conv(x)
        # flatten the output so it becomes a vector
        x = x.flatten(start_dim=1)

        return x

    def _compute_stride(self, input_length: int, kernal_size: int) -> int:
        """ Computes the stride parameter for time convolutions so that the time dimension
             of the input is almost wrapped after applying one convolution and one average pooling
             with given kernal_size
        """
        # the result is a root of the quadratic equation
        discriminant = (kernal_size - 1) ** 2 - 4 * (kernal_size - 1 - input_length)
        return int(
            (1 - kernal_size + math.sqrt(discriminant)) / 2
        )
    

class ResidualMlpProjector(nn.Module):
    """ Input is projected to output dim and then transforms as x = x + f(x)
         where f is MLP
    """
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
    ):
        super().__init__()

        self.linear_projector = nn.Linear(input_dim, output_dim)
        self.residual_layers = nn.Sequential(
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.Dropout(0.5)
        )
        self.norm_layer = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_projector(x)
        x = x + self.residual_layers(x)
        x = self.norm_layer(x)
        return x


class EEGEncoder(nn.Module):
    """ Encoder = Transformer layer + Spatio-time convolution + Residual MLP
    """
    def __init__(
            self,
            input_length: int,
            num_channels: int,
            output_dim: int = 1024,
            participants_embedding: nn.Embedding = None,
            conv_output_dim: int = 512,
            conv_kernal_size: int = 50,
            transformer_num_layers: int = 1,
            transformer_dim_feedforward: int = 2048,
            transformer_nhead: int = 1,
            transformer_dropout: float = 0.1,
            transformer_activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "relu"
    ):
        super().__init__()

        self.input_length = input_length
        self.num_channels = num_channels
        self.participants_embedding = participants_embedding

        self.transformer_stack = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                input_length,
                transformer_nhead,
                transformer_dim_feedforward,
                transformer_dropout,
                transformer_activation,
                batch_first=True
            )
            for _ in range(transformer_num_layers)
        ])

        self.conv_module = SpatioChannelConv(
            input_length,
            num_channels,
            conv_output_dim,
            conv_kernal_size
        )

        self.projector = ResidualMlpProjector(
            conv_output_dim,
            output_dim
        )

    def forward(self, x: torch.Tensor, participant_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Computes vector embeddings of the EEG signals.

        Args:
            x (torch.Tensor): eeg input with shape (batch_size, num_channels, input_length)
            participant_id (Optional[torch.Tensor], optional): participant's ids with shape 
                (batch_size). See class description for more details. Defaults to None.

        Returns:
            torch.Tensor: embeding vectors with shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # if we have participant's embeddings then add them as the first token to the EEG channels
        if self.participants_embedding is not None:
            # use participant's embeddings or mean embedding otherwise
            if participant_id is not None:
                participant_emb = self.participants_embedding(participant_id).unsqueeze(1)
            else:
                participant_emb = self.participants_embedding.weight.mean(
                    dim=0
                ).expand(
                    batch_size,
                    1,
                    self.input_length
                )
         
            x = torch.concat(
                [participant_emb, x],
                dim=1
            )

        x = self.transformer_stack(x)
        # leave participant's embedding if it exists
        if self.participants_embedding is not None:
            x = x[:, 1:, :]

        x = self.conv_module(x)
        x = self.projector(x)

        return x

# ------------------------------- EEG Net encoder ----------------------------------------

class EEGNet(nn.Module):
    def __init__(
            self,
            input_length: int,
            num_channels: int,
            output_dim: int,
            time_kernal_size: int = 64
    ):
        super().__init__()

        self.freq_filter_bank = nn.Conv2d(1, output_dim, kernel_size=(1, time_kernal_size))
        self.depthwise_spatial = nn.Conv2d(output_dim, output_dim, kernel_size=(num_channels, 1), groups=output_dim)
        # second time convolution fully wrap time dimension
        final_time_kernel_size = input_length - time_kernal_size + 1
        self.depthwise_freq_filter_bank = nn.Conv2d(output_dim, output_dim, kernel_size=(1, final_time_kernel_size), groups=output_dim)
        # final 1x1 conv along all filters
        self.mixture_conv = nn.Conv2d(output_dim, output_dim, kernel_size=(1, 1))

        self.transform = nn.Sequential(
            self.freq_filter_bank,
            self.depthwise_spatial,
            self.depthwise_freq_filter_bank,
            self.mixture_conv,
            nn.Flatten(start_dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add filter dimension to EEG
        x.unsqueeze_(1)

        return self.transform(x)


class ConvEEGEncoder(nn.Module):
    """ Encoder = EEG Net + Residual MLP
    """
    def __init__(
            self,
            input_length: int,
            num_channels: int,
            output_dim: int = 1024,
            participants_embedding: nn.Embedding = None,
            eeg_net_output_dim: int = 2048,
            eeg_net_time_kernal: int = 64
    ):
        super().__init__()

        self.input_length = input_length
        self.num_channels = num_channels
        self.participants_embedding = participants_embedding

        # num_channels is increased due to participant's token
        self.eeg_net = EEGNet(input_length, num_channels + 1, eeg_net_output_dim, eeg_net_time_kernal)

        self.projector = ResidualMlpProjector(
            eeg_net_output_dim,
            output_dim
        )

    def forward(self, x: torch.Tensor, participant_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Computes vector embeddings of the EEG signals.

        Args:
            x (torch.Tensor): eeg input with shape (batch_size, num_channels, input_length)
            participant_id (Optional[torch.Tensor], optional): participant's ids with shape 
                (batch_size). See class description for more details. Defaults to None.

        Returns:
            torch.Tensor: embeding vectors with shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # if we have participant's embeddings then add them as the first token to the EEG channels
        if self.participants_embedding is not None:
            # use participant's embeddings or mean embedding otherwise
            if participant_id is not None:
                participant_emb = self.participants_embedding(participant_id).unsqueeze(1)
            else:
                participant_emb = self.participants_embedding.weight.mean(
                    dim=0
                ).expand(
                    batch_size,
                    1,
                    self.input_length
                )
        else:
            # dummy participant's token if not provided
            participant_emb = torch.zeros((batch_size, 1, self.input_length), device=x.device)

        x = torch.concat(
                [participant_emb, x],
                dim=1
            )

        x = self.eeg_net(x)
        x = self.projector(x)

        return x
    

# ------------------------------- EEG encoders with loss for individual training ----------------------------------------


def contrastive_loss(image_features, brain_features, logit_scale) -> dict:
    # Normalize features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    brain_features = brain_features / brain_features.norm(dim=1, keepdim=True)

    # Compute cosine similarity as logits
    logit_scale = logit_scale.exp()
    logits_per_image = logit_scale * image_features @ brain_features.T
    logits_per_brain = logits_per_image.T

    # Compute cross-entropy loss
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=image_features.device, dtype=torch.long)

    total_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_brain, labels)) / 2

    return {'loss': total_loss, 'logits_per_brain': logits_per_brain}


class EEGEncoderWithLoss(EEGEncoder):
    def __init__(self, input_length, num_channels, output_dim = 1024, participants_embedding = None, conv_output_dim = 512, conv_kernal_size = 50, transformer_num_layers = 1, transformer_dim_feedforward = 2048, transformer_nhead = 1, transformer_dropout = 0.1, transformer_activation = "relu"):
        super().__init__(input_length, num_channels, output_dim, participants_embedding, conv_output_dim, conv_kernal_size, transformer_num_layers, transformer_dim_feedforward, transformer_nhead, transformer_dropout, transformer_activation)

        # Initialize logit_scale based on the original CLIP paper
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def loss(self, sub_ids, batch_eeg, image_features):
        """
        Computes the loss for the BrainEncoder.

        Args:
            sub_ids (Tensor): Participant IDs.
            batch_eeg (Tensor): Batch of EEG data.
            image_features (Tensor): Image features to compare with brain features.

        Returns:
            dict: A dictionary containing the loss and logits per brain.
        """
        # Get brain features
        brain_features = self(
            batch_eeg,
            sub_ids
        ).to(image_features.dtype)

        return contrastive_loss(image_features, brain_features, self.logit_scale)        

    @property
    def device(self):
        """
        Returns the device of the model parameters.

        Returns:
            torch.device: The device of the model parameters.
        """
        return next(iter(self.parameters()))[0].device
    

class ConvEEGEncoderWithLoss(ConvEEGEncoder):
    def __init__(self, input_length, num_channels, output_dim = 1024, participants_embedding = None, eeg_net_output_dim = 2048, eeg_net_time_kernal = 64):
        super().__init__(input_length, num_channels, output_dim, participants_embedding, eeg_net_output_dim, eeg_net_time_kernal)

        # Initialize logit_scale based on the original CLIP paper
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def loss(self, sub_ids, batch_eeg, image_features):
        """
        Computes the loss for the BrainEncoder.

        Args:
            sub_ids (Tensor): Participant IDs.
            batch_eeg (Tensor): Batch of EEG data.
            image_features (Tensor): Image features to compare with brain features.

        Returns:
            dict: A dictionary containing the loss and logits per brain.
        """
        # Get brain features
        brain_features = self(
            batch_eeg,
            sub_ids
        ).to(image_features.dtype)

        return contrastive_loss(image_features, brain_features, self.logit_scale)        

    @property
    def device(self):
        """
        Returns the device of the model parameters.

        Returns:
            torch.device: The device of the model parameters.
        """
        return next(iter(self.parameters()))[0].device