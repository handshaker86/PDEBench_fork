"""
Transolver: A Fast Transformer Solver for PDEs on General Geometries.

Adapted from https://github.com/thuml/Transolver for the PDEBench framework.
Implements Physics-Attention mechanism that computes attention on learned
physical state tokens rather than mesh points directly.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.models.layers import trunc_normal_
except ImportError:

    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        with torch.no_grad():
            tensor.normal_(mean, std)
            tensor.clamp_(a * std + mean, b * std + mean)
        return tensor


ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


# ---------------------------------------------------------------------------
# Utility modules
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        super().__init__()
        if act not in ACTIVATION:
            raise NotImplementedError(f"Activation '{act}' not supported")
        act_cls = ACTIVATION[act]
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_cls())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), act_cls()) for _ in range(n_layers)]
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ---------------------------------------------------------------------------
# Physics Attention modules
# ---------------------------------------------------------------------------


class Physics_Attention_Irregular_Mesh(nn.Module):
    """Physics-Attention for irregular meshes (1D, 2D, or 3D)."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, C = x.shape

        # (1) Slice
        fx_mid = (
            self.in_project_fx(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # B H N C
        x_mid = (
            self.in_project_x(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # B H N C
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / self.temperature
        )  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (
            (slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
        )

        # (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        # (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = out_x.permute(0, 2, 1, 3).reshape(B, N, -1)
        return self.to_out(out_x)


class Physics_Attention_Structured_Mesh_2D(nn.Module):
    """Physics-Attention for structured 2D meshes (uses Conv2d)."""

    def __init__(
        self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64, H=85, W=85, kernel=3
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W

        self.in_project_x = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, C = x.shape
        x = (
            x.reshape(B, self.H, self.W, C)
            .contiguous()
            .permute(0, 3, 1, 2)
            .contiguous()
        )  # B C H W

        # (1) Slice
        fx_mid = (
            self.in_project_fx(x)
            .permute(0, 2, 3, 1)
            .contiguous()
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # B H N C
        x_mid = (
            self.in_project_x(x)
            .permute(0, 2, 3, 1)
            .contiguous()
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # B H N C
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5)
        )  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (
            (slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
        )

        # (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        # (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = out_x.permute(0, 2, 1, 3).reshape(B, N, -1)
        return self.to_out(out_x)


class Physics_Attention_Structured_Mesh_3D(nn.Module):
    """Physics-Attention for structured 3D meshes (uses Conv3d)."""

    def __init__(
        self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=32, H=32, W=32, D=32, kernel=3
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W
        self.D = D

        self.in_project_x = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, C = x.shape
        x = (
            x.reshape(B, self.H, self.W, self.D, C)
            .contiguous()
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )  # B C H W D

        # (1) Slice
        fx_mid = (
            self.in_project_fx(x)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # B H N C
        x_mid = (
            self.in_project_x(x)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # B H N C
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5)
        )  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (
            (slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
        )

        # (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        # (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = out_x.permute(0, 2, 1, 3).reshape(B, N, -1)
        return self.to_out(out_x)


# ---------------------------------------------------------------------------
# Transolver blocks
# ---------------------------------------------------------------------------


class Transolver_block_1D(nn.Module):
    """Transolver encoder block for 1D / irregular meshes."""

    def __init__(
        self,
        num_heads,
        hidden_dim,
        dropout=0.0,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Transolver_block_2D(nn.Module):
    """Transolver encoder block for 2D structured meshes."""

    def __init__(
        self,
        num_heads,
        hidden_dim,
        dropout=0.0,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
        H=85,
        W=85,
        kernel=3,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Structured_Mesh_2D(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
            H=H,
            W=W,
            kernel=kernel,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Transolver_block_3D(nn.Module):
    """Transolver encoder block for 3D structured meshes."""

    def __init__(
        self,
        num_heads,
        hidden_dim,
        dropout=0.0,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
        H=32,
        W=32,
        D=32,
        kernel=3,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Structured_Mesh_3D(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
            H=H,
            W=W,
            D=D,
            kernel=kernel,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


# ---------------------------------------------------------------------------
# Weight initialization helpers
# ---------------------------------------------------------------------------


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def _reinit_orthogonal_slice(model):
    """Re-apply orthogonal init to slice projection weights after general init."""
    for m in model.modules():
        if hasattr(m, "in_project_slice"):
            torch.nn.init.orthogonal_(m.in_project_slice.weight)


# ---------------------------------------------------------------------------
# PDEBench-compatible Transolver models
# ---------------------------------------------------------------------------


class Transolver1d(nn.Module):
    """Transolver for 1D PDEs. Interface matches FNO1d: forward(x, grid)."""

    def __init__(
        self,
        num_channels,
        initial_step=10,
        prediction_step=1,
        n_hidden=256,
        n_layers=5,
        n_head=8,
        slice_num=32,
        mlp_ratio=1,
        dropout=0.0,
        act="gelu",
    ):
        super().__init__()
        self.num_channels = num_channels
        self.prediction_step = prediction_step
        self.n_hidden = n_hidden

        space_dim = 1
        fun_dim = initial_step * num_channels
        self.preprocess = MLP(
            fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act
        )

        self.blocks = nn.ModuleList(
            [
                Transolver_block_1D(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    out_dim=prediction_step * num_channels,
                    slice_num=slice_num,
                    last_layer=(i == n_layers - 1),
                )
                for i in range(n_layers)
            ]
        )

        self.placeholder = nn.Parameter(
            (1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float)
        )

        self.apply(_init_weights)
        _reinit_orthogonal_slice(self)

    def forward(self, x, grid):
        # x: [B, L, initial_step * num_channels]
        # grid: [L, 1]
        B, L, _ = x.shape
        grid_expanded = grid.unsqueeze(0).expand(B, -1, -1)  # [B, L, 1]

        fx = torch.cat([x, grid_expanded], dim=-1)  # [B, L, fun_dim + 1]
        fx = self.preprocess(fx)
        fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            fx = block(fx)

        # fx: [B, L, prediction_step * num_channels]
        return fx.view(B, L, self.prediction_step, self.num_channels)


class Transolver2d(nn.Module):
    """Transolver for 2D PDEs. Interface matches FNO2d: forward(x, grid)."""

    def __init__(
        self,
        num_channels,
        initial_step=10,
        prediction_step=1,
        n_hidden=256,
        n_layers=5,
        n_head=8,
        slice_num=32,
        mlp_ratio=1,
        dropout=0.0,
        act="gelu",
        kernel=3,
        H=64,
        W=64,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.prediction_step = prediction_step
        self.n_hidden = n_hidden
        self.H = H
        self.W = W

        space_dim = 2
        fun_dim = initial_step * num_channels
        self.preprocess = MLP(
            fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act
        )

        self.blocks = nn.ModuleList(
            [
                Transolver_block_2D(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    out_dim=prediction_step * num_channels,
                    slice_num=slice_num,
                    H=H,
                    W=W,
                    kernel=kernel,
                    last_layer=(i == n_layers - 1),
                )
                for i in range(n_layers)
            ]
        )

        self.placeholder = nn.Parameter(
            (1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float)
        )

        self.apply(_init_weights)
        _reinit_orthogonal_slice(self)

    def forward(self, x, grid):
        # x: [B, Lx, Ly, initial_step * num_channels]
        # grid: [Lx, Ly, 2]
        B, Lx, Ly, _ = x.shape
        N = Lx * Ly

        x_flat = x.reshape(B, N, -1)  # [B, N, fun_dim]
        grid_flat = grid.reshape(1, N, -1).expand(B, -1, -1)  # [B, N, 2]

        fx = torch.cat([x_flat, grid_flat], dim=-1)  # [B, N, fun_dim + 2]
        fx = self.preprocess(fx)
        fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            fx = block(fx)

        # fx: [B, N, prediction_step * num_channels]
        return fx.view(B, Lx, Ly, self.prediction_step, self.num_channels)


class Transolver3d(nn.Module):
    """Transolver for 3D PDEs. Interface matches FNO3d: forward(x, grid)."""

    def __init__(
        self,
        num_channels,
        initial_step=10,
        prediction_step=1,
        n_hidden=256,
        n_layers=5,
        n_head=8,
        slice_num=32,
        mlp_ratio=1,
        dropout=0.0,
        act="gelu",
        kernel=3,
        H=32,
        W=32,
        D=32,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.prediction_step = prediction_step
        self.n_hidden = n_hidden
        self.H = H
        self.W = W
        self.D = D

        space_dim = 3
        fun_dim = initial_step * num_channels
        self.preprocess = MLP(
            fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act
        )

        self.blocks = nn.ModuleList(
            [
                Transolver_block_3D(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    out_dim=prediction_step * num_channels,
                    slice_num=slice_num,
                    H=H,
                    W=W,
                    D=D,
                    kernel=kernel,
                    last_layer=(i == n_layers - 1),
                )
                for i in range(n_layers)
            ]
        )

        self.placeholder = nn.Parameter(
            (1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float)
        )

        self.apply(_init_weights)
        _reinit_orthogonal_slice(self)

    def forward(self, x, grid):
        # x: [B, Lx, Ly, Lz, initial_step * num_channels]
        # grid: [Lx, Ly, Lz, 3]
        B, Lx, Ly, Lz, _ = x.shape
        N = Lx * Ly * Lz

        x_flat = x.reshape(B, N, -1)  # [B, N, fun_dim]
        grid_flat = grid.reshape(1, N, -1).expand(B, -1, -1)  # [B, N, 3]

        fx = torch.cat([x_flat, grid_flat], dim=-1)  # [B, N, fun_dim + 3]
        fx = self.preprocess(fx)
        fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            fx = block(fx)

        # fx: [B, N, prediction_step * num_channels]
        return fx.view(B, Lx, Ly, Lz, self.prediction_step, self.num_channels)
