import torch
from torch import nn


config_1 = {
    "input_length" : 500,
    "num_channels" : 63,
    "output_dim" : 1024,
    "conv_output_dim": 1024,
    "conv_kernal_size": 100
}

config_2 = {
    "participants_embedding": nn.Embedding(12, config_1["input_length"])
}
config_2.update(config_1)

config_3 = config_1.copy()
config_3.update({
    "conv_output_dim": 512,
    "conv_kernal_size": 20
} )

configs = [config_1, config_2, config_3]