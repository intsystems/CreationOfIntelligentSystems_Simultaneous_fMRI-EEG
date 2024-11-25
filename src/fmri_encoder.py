import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re

class PatchEmbed1D(nn.Module):
    def __init__(self, latent_dim=4096, patch_size=16, in_chans=1, embed_dim=1024):
        super().__init__()
        num_patches = latent_dim // patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).transpose(1, 2).contiguous()
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class fMRIEncoder(nn.Module):
    def __init__(self, latent_dim=4096, patch_size=16, embed_dim=1024, in_chans=1, depth=12, num_heads=8, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed1D(latent_dim, patch_size, in_chans, embed_dim)
        num_patches = latent_dim // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, drop=dropout, attn_drop=dropout, drop_path=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]  # Return only the CLS token
    

class RidgeRegression(nn.Module):
    def __init__(self, masks_path, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.masks = self.load_masks_from_directory(masks_path)

        input_sizes = [int(torch.sum(mask)) for mask in self.masks]
        self.linears = nn.ModuleList([
            nn.Linear(input_size, out_dim) for input_size in input_sizes
        ])

    def load_masks_from_directory(self, directory_path):
        """
        Loads all mask files from the given directory and combines them into a single tensor.

        :param directory_path: Path to the directory containing mask files
        :return: Tensor containing all masks
        """
        masks = []
        files = sorted(os.listdir(directory_path), key=lambda x: int(re.search(r'sub-(\d+)\.pth', x).group(1)))
        # Iterate over the files and load the masks
        for file_name in files:
            if file_name.startswith("sub-") and file_name.endswith(".pth"):
                file_path = os.path.join(directory_path, file_name)
                mask = torch.load(file_path)
                masks.append(mask)
        masks_tensor = torch.stack(masks)

        return masks_tensor

    def apply_mask_and_flatten(self, fmri_batch, mask):
        """
        Applies a mask to a batch of FMRI tensors and returns a batch of vectors.

        :param fmri_batch: Tensor of shape (batch_size, depth, height, width)
        :param mask: Tensor of shape (depth, height, width) with values 0 and 1
        :return: Tensor of shape (batch_size, num_masked_voxels)
        """
        # Flatten the tensor and keep only the active voxels
        flattened_batch = fmri_batch.reshape(fmri_batch.size(0), -1)  # Reshape to (batch_size, depth * height * width)
        # Keep only the active voxels
        active_voxels = flattened_batch[:, mask.view(-1) == 1]

        return active_voxels

    def generate_sub_batches(self, id_batch, fmri_batch):
        """
        Generates sub-batches for each unique ID in the id_batch.

        :param fmri_batch: Tensor of shape (batch_size, depth, height, width)
        :param id_batch: Tensor of shape (batch_size,) containing IDs
        :return: Dictionary with IDs as keys and corresponding sub-batches as values
        """
        # Initialize an empty dictionary to store the sub-batches
        sub_batches = {}
        # Get unique IDs
        unique_ids = id_batch.unique()
        # Initialize a list to store the original indices
        original_indices = []
        # Iterate over unique IDs and create sub-batches
        for unique_id in unique_ids:
            # Find indices of the current ID in the id_batch
            indices = (id_batch == unique_id).nonzero(as_tuple=True)[0]
            # Create the sub-batch for the current ID
            sub_batch = fmri_batch[indices]
            # Store the sub-batch in the dictionary
            sub_batches[unique_id.item()] = sub_batch
            # Store the original indices
            original_indices.extend(indices.tolist())

        return sub_batches, original_indices

    def forward(self, id_batch, fmri_batch):
        sub_batches, original_indices = self.generate_sub_batches(id_batch, fmri_batch)
        outputs = []
        for sub_num, sub_batch in sub_batches.items():
            sub_batch = self.apply_mask_and_flatten(sub_batch, self.masks[sub_num].to(sub_batch.device))
            out = self.linears[sub_num](sub_batch)
            outputs.append(out)
        # Concatenate the outputs along the batch dimension
        outputs = torch.cat(outputs, dim=0)
        # Reorder the outputs based on the original indices
        outputs = outputs[torch.argsort(torch.tensor(original_indices))]
        # Reshape the output to [batch_size, 1, out_features]
        outputs = outputs.view(-1, 1, self.out_dim)
        return outputs