import pytest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.eeg_encoder import SpatioChannelConv, ResidualMlpProjector, EEGEncoder
from src.eeg_encoder import ConvEEGEncoder
from tests.eeg_encoder_configs import encoder_configs, conv_encoder_configs


# setup encoders's configs for the tests
@pytest.fixture(scope="function", params=encoder_configs)
def model_config_setup(request):
    return request.param

# setup conv_encoders's configs for the tests
@pytest.fixture(scope="function", params=conv_encoder_configs)
def conv_model_config_setup(request):
    return request.param


def test_spatio_channel_conv_shapes(model_config_setup):
    """ tests right output shapes of the encoder's convolution module
    """
    config: dict = model_config_setup

    net = SpatioChannelConv(
        config["input_length"],
        config["num_channels"],
        config["conv_output_dim"],
        config["conv_kernal_size"]
    )

    # check final stride for the convolutions
    print("Stride =", net._compute_stride(config["input_length"],  config["conv_kernal_size"]))
    print("Kernal =", config["conv_kernal_size"])

    model_input = torch.rand((2, config["num_channels"], config["input_length"]))
    model_output = net(model_input)
    assert model_output.size() == torch.Size([2, config["conv_output_dim"]])

def test_encoder_shapes(model_config_setup):
    """ tests right output shapes of the whole encoder
    """
    config: dict = model_config_setup

    net = EEGEncoder(**config)

    model_input = torch.rand((2, config["num_channels"], config["input_length"]))
    model_output = net(model_input)
    assert model_output.size() == torch.Size([2, config["output_dim"]])

def test_encoder_grads(model_config_setup):
    """ tests grad population of the backward pass on the encoder
    """
    config: dict = model_config_setup

    net = EEGEncoder(**config)

    model_input = torch.rand((2, config["num_channels"], config["input_length"]))
    model_output = net(model_input)
    targets = torch.rand((2, config["output_dim"]))
    loss = F.mse_loss(model_output, targets)
    loss.backward()

    net_grads = torch.concat([
       param.grad.flatten() for param in net.parameters()
    ])

    print("Mean grad value=", net_grads.mean().item())

    assert not torch.allclose(net_grads, torch.zeros_like(net_grads), atol=1e-3)

def test_conv_encoder_shapes(conv_model_config_setup):
    """ tests right output shapes of the whole encoder
    """
    config: dict = conv_model_config_setup

    net = ConvEEGEncoder(**config)

    model_input = torch.rand((2, config["num_channels"], config["input_length"]))
    model_output = net(model_input)
    assert model_output.size() == torch.Size([2, config["output_dim"]])

def test_conv_encoder_grads(conv_model_config_setup):
    """ tests grad population of the backward pass on the encoder
    """
    config: dict = conv_model_config_setup

    net = ConvEEGEncoder(**config)

    model_input = torch.rand((2, config["num_channels"], config["input_length"]))
    model_output = net(model_input)
    targets = torch.rand((2, config["output_dim"]))
    loss = F.mse_loss(model_output, targets)
    loss.backward()

    net_grads = torch.concat([
       param.grad.flatten() for param in net.parameters()
    ])

    print("Mean grad value=", net_grads.mean().item())

    assert not torch.allclose(net_grads, torch.zeros_like(net_grads), atol=1e-3)

