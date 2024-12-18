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
            output_dim: int = None,
            kernal_size: int = None
    ):
        super().__init__()

        self.time_conv = nn.Sequential(
            nn.Conv2d(1, 20, (1 ,25), (1, 1)),
            nn.AvgPool2d((1 ,51), (1, 5))
        )

        self.channel_conv = nn.Sequential(
            nn.BatchNorm2d(20),
            nn.ELU(),
            nn.Conv2d(20, 20, (num_channels, 1)),
            nn.ELU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ The model is oriented around precise input_length and num channels of our task.
            After the time conv we have untouched chaneel dimension and time dim = 90.
            After the channel conv we have untouched time dim and fully convolved channel dim.
            The final output dim = 1800.
        """
        # add feature maps dimension
        x.unsqueeze_(dim=1)
        # apply time convolutions
        x = self.time_conv(x)
        # apply channel convolutions
        x = self.channel_conv(x)
        # flatten the output so it becomes a vector
        x = x.flatten(start_dim=1)

        return x


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
            conv_output_dim: int = 1800,
            conv_kernal_size: int = None,
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
