import torch
from torch import nn
import torch.nn.functional as F


class EegFmriWeightFuser(nn.Module):
    """Fusing embeddings using their convex combination.
        The combination's coefficents are therefore interpretable.
    """
    def __init__(
            self
        ):
        super().__init__()

        self.eeg_weight = nn.Parameter(torch.FloatTensor([1.]))
        self.fmri_weight = nn.Parameter(torch.FloatTensor([1.]))

    def forward(self, eeg_emb: torch.Tensor, fmri_emb: torch.Tensor) -> torch.Tensor:
        norming_factor = self.eeg_weight + self.fmri_weight

        return (self.eeg_weight * eeg_emb + self.fmri_weight * fmri_emb) / norming_factor
    
class EegFmriMlpFuser(nn.Module):
    """Fusing embeddings using residual MLP
    """
    def __init__(
            self,
            emb_dim: int
        ):
        super().__init__()

        self.residual_layer = nn.Sequential(*[
            nn.Linear(2*emb_dim, 2*emb_dim),
            nn.ELU()
        ])

        self.linear_projector = nn.Linear(2*emb_dim, emb_dim)

    def forward(self, eeg_emb: torch.Tensor, fmri_emb: torch.Tensor) -> torch.Tensor:
        uni_emb = torch.concat([eeg_emb, fmri_emb])
        uni_emb = uni_emb + self.residual_layer(uni_emb)
        uni_emb = self.linear_projector(uni_emb)

        return uni_emb

