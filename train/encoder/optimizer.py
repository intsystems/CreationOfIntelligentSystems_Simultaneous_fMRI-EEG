""" define your optimizer and connected stuff here to be used in training
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

def get_optimizer(model: nn.Module) -> optim.Optimizer:
    LR = 1e-3
    
    return optim.Adam(model.parameters(), lr=LR)