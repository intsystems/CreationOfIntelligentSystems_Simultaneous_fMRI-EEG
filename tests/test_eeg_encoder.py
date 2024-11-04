import pytest
from src.egg_encoder import *
from model_configs import configs


# setup model's config for the tests
@pytest.fixture(scope="function", params=configs)
def model_config_setup(request):
    return request.param

def test_spatio_channel_conv_shapes(model_config_setup):
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


