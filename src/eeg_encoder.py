from typing import Union, Callable, Optional
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class SpatioChannelConv(nn.Module):
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
             of the input is almost wrapped after applying one convolution and one avr. pooling
             with given kernal_size
        """
        # the result is a root of the quadratic equation
        discriminant = (kernal_size - 1) ** 2 - 4 * (kernal_size - 1 - input_length)
        return int(
            (1 - kernal_size + math.sqrt(discriminant)) / 2
        )
    

class ResidualMlpProjector(nn.Module):
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
        if self.participants_embedding:
            # use participant's embeddings or mean embedding otherwise
            if participant_id:
                participant_emb = self.participants_embedding(participant_id)
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
        if self.participants_embedding:
            x = x[:, 1:, :]

        x = self.conv_module(x)
        x = self.projector(x)

        return x



