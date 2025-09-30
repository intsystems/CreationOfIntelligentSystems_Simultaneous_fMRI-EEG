import math
import torch
from torch import nn


@torch.autocast(device_type="cuda", enabled=False)
def get_freqs(dim, max_period=10000.0):
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=dim, dtype=torch.float32)
        / dim
    )
    return freqs


@torch.autocast(device_type="cuda", dtype=torch.float32)
def apply_scale_shift_norm(norm, x, scale, shift):
    return (1.0 + scale) * norm(x) + shift


@torch.autocast(device_type="cuda", dtype=torch.float32)
def apply_gate_sum(x, out, gate):
    return x + gate * out


class TimeEmbeddings(nn.Module):

    def __init__(self, model_dim, time_dim, max_period=10000.0):
        super().__init__()
        assert model_dim % 2 == 0
        self.model_dim = model_dim
        self.max_period = max_period
        self.register_buffer(
            "freqs",
            get_freqs(model_dim // 2, max_period),
            persistent=False,
        )
        self.in_layer = nn.Linear(model_dim, time_dim, bias=True)
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, time_dim, bias=True)

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, time):
        args = torch.outer(time, self.freqs.to(device=time.device))
        time_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        time_embed = self.out_layer(self.activation(self.in_layer(time_embed)))
        return time_embed


class Modulation(nn.Module):

    def __init__(self, time_dim, model_dim, num_params):
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, num_params * model_dim)
        self.out_layer.weight.data.zero_()
        self.out_layer.bias.data.zero_()

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, x):
        return self.out_layer(self.activation(x))


class FeedForward(nn.Module):

    def __init__(self, dim, ff_dim):
        super().__init__()
        self.in_layer = nn.Linear(dim, ff_dim, bias=False)
        self.activation = nn.GELU()
        self.out_layer = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x):
        return self.out_layer(self.activation(self.in_layer(x)))


class FiLMResidualBlock(nn.Module):
    # https://arxiv.org/abs/1709.07871

    def __init__(self, model_dim, time_dim, ff_dim):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)
        self.modulation = Modulation(time_dim, model_dim, 3)

    def forward(self, x, time_embed):
        shift, scale, gate = torch.chunk(self.modulation(time_embed), 3, dim=-1)
        y = apply_scale_shift_norm(self.norm, x, scale, shift).type_as(x)
        y = self.feed_forward(y)
        x = apply_gate_sum(x, y, gate).type_as(x)
        return x


class FlowMatchingModel(nn.Module):

    def __init__(
        self,
        latent_dim=1024,
        time_dim=512,
        model_dim=2048,
        ff_dim=5120,
        num_blocks=12,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.model_dim = model_dim

        self.time_embeddings = TimeEmbeddings(model_dim, time_dim)
        self.in_proj = nn.Linear(latent_dim, model_dim)

        self.blocks = nn.ModuleList(
            [FiLMResidualBlock(model_dim, time_dim, ff_dim) for _ in range(num_blocks)]
        )

        self.out_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.out_proj = nn.Linear(model_dim, latent_dim)

    def forward(self, x, time):
        time_embed = self.time_embeddings(time)
        h = self.in_proj(x)

        for block in self.blocks:
            h = block(h, time_embed)

        h = self.out_norm(h).type_as(x)
        out = self.out_proj(h)
        return out


if __name__ == "__main__":
    model = FlowMatchingModel()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
