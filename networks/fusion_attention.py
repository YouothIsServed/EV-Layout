import torch
from torch import nn
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
class Fusion_attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.image_size = (64, 64)
        self.patch_counts = 2048
        self.patch_size = (8, 8)
        self.channels = 3
        self.dim = dim

        image_height, image_width = pair(self.image_size)
        patch_height, patch_width = pair(self.patch_size)

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = self.channels * patch_height * patch_width

        self.to_patch_embedding_x = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.dim),
            nn.LayerNorm(self.dim),
        )
        self.to_patch_embedding_time_map = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.dim),
            nn.LayerNorm(self.dim),
        )

        self.norm_x = nn.LayerNorm(dim)
        self.norm_time_map = nn.LayerNorm(dim)

        self.to_qkv_x = nn.Linear(dim, 64 * 3, bias=False)
        self.to_qkv_time_map = nn.Linear(dim, 64 * 3, bias=False)

        self.attend = nn.Softmax(dim=-1)

        self.to_out = nn.Sequential(
            nn.Linear(64, dim),
            nn.Linear(dim, patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=8, w=8, p1=patch_height, p2=patch_width)
        )

        self.scale = 64 ** -0.5

    def forward(self, x, time_map):
        x = self.to_patch_embedding_x(x)
        x = self.norm_x(x)
        x = self.to_qkv_x(x).chunk(3, dim=-1)
        q_x, k_x, v_x = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=1), x)

        time_map = self.to_patch_embedding_time_map(time_map)
        time_map = self.norm_time_map(time_map)
        time_map = self.to_qkv_time_map(time_map).chunk(3, dim=-1)
        q_t, k_t, v_t = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=1), time_map)

        dots = torch.matmul(q_x, k_t.transpose(-1, -2))*self.scale
        dots = self.attend(dots)
        out = torch.matmul(dots, v_x)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

if __name__ == "__main__":
    a = torch.ones(64, 3, 64, 64)
    b = torch.ones(64, 3, 64, 64)
    Fusion_atten = Fusion_attention(128)
    out = Fusion_atten(a, b)