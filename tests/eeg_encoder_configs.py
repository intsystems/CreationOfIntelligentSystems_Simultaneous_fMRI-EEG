""" eeg encoder configs for tests
"""
import torch
from torch import nn


config_1 = {
    "input_length" : 525,
    "num_channels" : 63,
    "output_dim" : 1024,
    "conv_output_dim": 1820,
    "conv_kernal_size": None
}

config_2 = {
    "participants_embedding": nn.Embedding(12, config_1["input_length"])
}
config_2.update(config_1)

# config_3 = config_1.copy()
# config_3.update({
#     "conv_output_dim": 512,
#     "conv_kernal_size": 20
# } )

config_4 = {
    "input_length" : 525,
    "num_channels" : 61,
    "output_dim" : 1024,
    "conv_output_dim": 1820,
    "conv_kernal_size": None,
    "transformer_num_layers": 0
}

encoder_configs = [config_1, config_2, config_4]


conv_config_1 = {
    "input_length" : 525,
    "num_channels" : 63,
    "output_dim" : 1024,
    "eeg_net_time_kernal": 120,
}

conv_config_2 = {
    "participants_embedding": nn.Embedding(12, config_1["input_length"])
}
conv_config_2.update(conv_config_1)

conv_config_3 = {
    "eeg_net_time_kernal": 64,
    "eeg_net_output_dim": 1500
}
conv_config_3.update(conv_config_2)

conv_encoder_configs = [conv_config_1, conv_config_2, conv_config_3]
