import time
import torch
import torch.nn as nn


class Timer:
    def __init__(self, name=''):
        self.start_time = 0
        self.end_time = 0
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        print(f"Elapsed time {self.name}: {self.end_time - self.start_time:.2f} seconds")
        

def get_model_size_mb(model_module: nn.Module) -> float:
    """
    Calculate the size of a PyTorch model or module in MB.

    Args:
        model_module (nn.Module): The model or module to calculate the size for.

    Returns:
        float: The size of the model or module in MB.
    """
    if not isinstance(model_module, nn.Module):
        raise ValueError("The input must be an instance of torch.nn.Module")

    param_size = sum(p.nelement() * p.element_size() for p in model_module.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model_module.buffers())

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def get_tensor_size_mb(tensor: torch.Tensor) -> float:
    """
    Calculate the size of a PyTorch tensor in MB.

    Args:
        tensor (torch.Tensor): The tensor to calculate the size for.

    Returns:
        float: The size of the tensor in MB.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("The input must be an instance of torch.Tensor")

    size_bytes = tensor.nelement() * tensor.element_size()
    size_mb = size_bytes / 1024**2
    return size_mb