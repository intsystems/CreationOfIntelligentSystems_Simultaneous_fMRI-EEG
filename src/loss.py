import torch
from torch.nn import functional as F
from torch import nn as nn

class ClipLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_features, brain_features, logit_scale):
        logits_per_image = logit_scale * image_features @ brain_features.T
        logits_per_brain = logit_scale * brain_features @ image_features.T

        num_logits = logits_per_image.shape[0]
        labels = torch.arange(num_logits, device=image_features.device, dtype=torch.long)

        total_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_brain, labels)) / 2
        return total_loss
