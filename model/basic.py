import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def convbnrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class CrossFusionBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, y):
        # x: query, y: key/value
        x_norm = self.norm1(x)
        y_norm = self.norm2(y)
        # Q, K, V
        out, _ = self.attn(x_norm, y_norm, y_norm)
        out = x + out  # residual
        out = out + self.ffn(self.norm3(out))
        return out

class FusionBlock(nn.Module):
    def __init__(self, dim):
        super(FusionBlock, self).__init__()
        self.reluconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

    def forward(self, part_a, part_b):
        fusion = part_a + part_b
        fusion1 = self.reluconv(fusion)
        fusion_upsample = F.interpolate(fusion + fusion1, scale_factor=2.0, mode='bilinear', align_corners=True)
        output = self.conv(fusion_upsample)
        return output


class SinusoidalPositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_h=100, max_w=100):
        super(SinusoidalPositionalEncoding2D, self).__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D encoding (H: d/2, W: d/2)")

        self.d_model = d_model
        self.max_h = max_h
        self.max_w = max_w

        pe_h = self._create_1d_enc(d_model // 2, max_h)  # (d/2, max_h)
        pe_w = self._create_1d_enc(d_model // 2, max_w)  # (d/2, max_w)

        pe_h = pe_h.unsqueeze(2).expand(-1, -1, max_w)  # (d/2, H, W)
        pe_w = pe_w.unsqueeze(1).expand(-1, max_h, -1)  # (d/2, H, W)

        pe = torch.cat([pe_h, pe_w], dim=0)  # (d_model, H, W)
        pe = pe.unsqueeze(0)  # (1, d_model, H, W)

        self.register_buffer('pe', pe)

    def _create_1d_enc(self, d, max_len):
        pe = torch.zeros(d, max_len)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))  # (d/2,)

        pe[0::2, :] = torch.sin(position * div_term.unsqueeze(1))  # even indices
        pe[1::2, :] = torch.cos(position * div_term.unsqueeze(1))  # odd indices
        return pe

    def forward(self, x):
        _, _, H, W = x.shape
        if H > self.max_h or W > self.max_w:
            raise ValueError(f"Input size ({H}x{W}) exceeds max size ({self.max_h}x{self.max_w}).")
        return x + self.pe[:, :, :H, :W]
    