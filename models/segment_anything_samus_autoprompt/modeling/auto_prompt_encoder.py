# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from re import X
from tkinter import N
from tokenize import Double
import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type

from .common import LayerNorm2d, softmax_one
from einops import rearrange


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

class vitAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, qx, kx):
        q = self.to_q(qx)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        kv = self.to_kv(kx).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn =  softmax_one(dots, dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2in(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x1, x2, **kwargs):
        return self.fn(self.norm1(x1), self.norm2(x2), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm2in(dim, vitAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x1, x2):
        for attn, ff in self.layers:
            ax = attn(x1, x2)
            x1 = ax + x1
            x1 = ff(x1) + x1
        return x1


class CrossTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm2in(dim, vitAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm2in(dim, vitAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x1, x2):
        for attn1, attn2, ff1, ff2 in self.layers:
            ax1, ax2 = attn1(x1, x2), attn2(x2, x1)
            x1, x2 = ax1 + x1, ax2 + x2
            x1 = ff1(x1) + x1
            x2 = ff2(x2) + x2
        return x1, x2


class Prompt_Embedding_Generator(nn.Module):
    def __init__(
        self,
        out_dim: int = 256,
        base_dim: int=48,
        num_heads: int = 8,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.embed_dim = out_dim
        self.base_dim = base_dim
        self.num_heads = num_heads
        self.scale = (out_dim//self.num_heads)**-0.5

        self.object_token = nn.Parameter(torch.randn(1, 50, self.embed_dim))
        self.cross_token_token = CrossTransformer(dim=self.embed_dim, depth=2, heads=8, dim_head=64)
        self.token_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.cross_image_token = CrossTransformer(dim=self.embed_dim, depth=2, heads=8, dim_head=64)
        
    def forward(self,
        img_embedding: torch.Tensor,
        output_token: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returning new_img_embedding, new_output_token, and object_token

        Arguments:
          img_embedding: torch.Tensor with shape (B, embed_dim, 32, 32)
          output_token: torch.Tensor with shape (B, 5, 256)

        Returns:
          torch.Tensor: new img embedding, with shape (B, embed_dim, 16, 16).
          torch.Tensor: new ouput token, with shape (B, 5, 256).
          torch.Tensor: object token, with shape Bx1x(embed_dim).
        """
        #img_embedding = self.feature_adapter(img_embedding)
        b, c, h, w = img_embedding.shape
        img_embedding = rearrange(img_embedding, 'b c h w -> b (h w) c')
        object_token, new_output_token = self.cross_token_token(self.object_token, output_token)
        object_token = self.token_proj(object_token) + self.object_token
        new_output_token = self.token_proj(new_output_token) + output_token
        tokens = torch.cat([object_token, output_token], dim=1) # [b 6 d]
        new_img_embedding, tokens = self.cross_image_token(img_embedding, tokens) 
        new_img_embedding = rearrange(new_img_embedding, 'b (h w) c -> b c h w', h=h)
        return new_img_embedding, tokens[:, :1, :], tokens[:, 1:, :]


class MaskAttention(nn.Module):
    def __init__(self, embedding=256, kernel_size=7):
        super(MaskAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv = nn.Conv2d(3, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(embedding, embedding, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(embedding, embedding, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(embedding, 1, kernel_size=3, padding=1, bias=False)
        self.convup1 = nn.Conv2d(1, embedding//2, kernel_size=3, padding=1, bias=False)
        self.convup2 = nn.Conv2d(embedding//2, embedding, kernel_size=3, padding=1, bias=False)
 
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1(x1)
        x3 = self.conv3(x1)
    
        avg_attn = torch.mean(x2, dim=1, keepdim=True)
        max_attn, _ = torch.max(x2, dim=1, keepdim=True)
        attn = torch.cat([avg_attn, max_attn, x3], dim=1)
        attn = self.conv(attn)
        attn1 = self.sigmoid(attn)
        x_up = self.convup1(attn)
        x_up = self.convup2(x_up)
        x_up = attn1 * x_up

        return x_up + x1, attn


def pos_neg_clicks(mask, class_id=1, pos_prompt_number=1, neg_prompt_number=1):
    pos_indices = np.argwhere(mask == class_id)
    pos_indices[:, [0,1]] = pos_indices[:, [1,0]]
    pos_label = 1
    if len(pos_indices) == 0:
        pos_label = -1 # or 0
        pos_indices = np.argwhere(mask != class_id)
        pos_indices[:, [0,1]] = pos_indices[:, [1,0]]
    pos_num = min(len(pos_indices), pos_prompt_number)
    pos_prompt_indices = np.random.randint(len(pos_indices), size=pos_num)
    pos_prompt = pos_indices[pos_prompt_indices]
    pos_label = np.repeat(pos_label, pos_num)

    neg_indices = np.argwhere(mask != class_id)
    neg_indices[:, [0,1]] = neg_indices[:, [1,0]]
    neg_num = pos_prompt_number + neg_prompt_number - pos_num
    neg_prompt_indices = np.random.randint(len(neg_indices), size=neg_num)
    neg_prompt = neg_indices[neg_prompt_indices]
    neg_label = np.repeat(0, neg_num)

    pt = np.vstack((pos_prompt, neg_prompt))
    point_label = np.hstack((pos_label, neg_label))
    return pt, np.array(point_label)

def make_prompt_from_mask(mask):
    pts, point_labels = [], []
    with torch.no_grad():
        predict = torch.sigmoid(mask)
        predict = predict.detach().cpu().numpy()  
        seg = predict[:, 0, :, :] > 0.5 
        for i in range(seg.shape[0]):
            pt, point_label = pos_neg_clicks(seg[i, :, :], pos_prompt_number=10, neg_prompt_number=0)
            pts.append(pt[None, :, :])
            point_labels.append(point_label[None, :])
        pts = np.concatenate(pts, axis=0)
        point_labels = np.concatenate(point_labels, axis=0)
    coords_torch = torch.as_tensor(pts, dtype=torch.float32, device=mask.device)
    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=mask.device)
    if len(pts.shape) == 2:
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
    pts = (coords_torch, labels_torch)
    return pts