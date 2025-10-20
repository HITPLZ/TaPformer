import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp
from src.base.model import BaseModel

# ----------------------
# Multi-head Attention with graph bias support
# ----------------------
class Attention(nn.Module):
    """
    Multi-head self-attention supporting attn_bias and attn_mask.
    Input shape: [B, N, D]
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None, attn_mask=None):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        if attn_bias is not None:
            attn = attn + attn_bias
        if attn_mask is not None:
            # attn_mask: [B, N, N] or [B,1,N,N]
            mask = attn_mask.unsqueeze(1) if attn_mask.dim()==3 else attn_mask
            attn = attn.masked_fill(mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

# ----------------------
# Window-attention block with graph bias instead of GCN
# ----------------------
class WindowAttBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, P, N, mlp_ratio=4.0):
        super().__init__()
        D, H, M = hidden_size, num_heads, int(hidden_size * mlp_ratio)
        self.P, self.N = P, N
        # depth attention sublayer
        self.depth_norm1 = nn.LayerNorm(D)
        self.depth_attn = Attention(D, num_heads=H, qkv_bias=True,
                                   attn_drop=0.1, proj_drop=0.1)
        self.depth_norm2 = nn.LayerNorm(D)
        self.depth_mlp = Mlp(in_features=D, hidden_features=M,
                             act_layer=nn.GELU, drop=0.1)
        # patch-to-patch attention sublayer
        self.breadth_norm1 = nn.LayerNorm(D)
        self.breadth_attn = Attention(D, num_heads=H, qkv_bias=True,
                                      attn_drop=0.1, proj_drop=0.1)
        self.breadth_norm2 = nn.LayerNorm(D)
        self.breadth_mlp = Mlp(in_features=D, hidden_features=M,
                               act_layer=nn.GELU, drop=0.1)

    def forward(self, x, mask, patch_adj):
        B, T, PN, D = x.shape
        P, N = self.P, self.N
        assert PN == P * N
        # reshape to [B, T, P, N, D]
        x = x.view(B, T, P, N, D)
        m = mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1,1,P,N,1]
        # 1) Depth Attention with graph bias
        y = x.view(B * T * P, N, D)
        y = self.depth_norm1(y)
        # build attn_bias from patch_adj weights
        A = patch_adj  # [P, N, N]
        A_flat = A.unsqueeze(0).unsqueeze(0)            \
                 .expand(B, T, P, N, N)               \
                 .contiguous()                         \
                 .view(B * T * P, N, N)
        bias = torch.log(A_flat + 1e-6).unsqueeze(1)     # [B*T*P,1,N,N]
        y = self.depth_attn(y, attn_bias=bias)
        y = y.view(B, T, P, N, D)
        x = x + y * m
        # depth MLP
        y = x.view(B * T * P, N, D)
        y = self.depth_norm2(y)
        y = self.depth_mlp(y)
        x = x + (y.view(B, T, P, N, D) * m)
        # 2) patch-to-patch attention
        z = x * m
        z = z.sum(dim=3) / (m.sum(dim=3).clamp(min=1))  # [B, T, P, D]
        z = z.view(B * T, P, D)
        z = self.breadth_norm1(z)
        z = self.breadth_attn(z)
        z = self.breadth_mlp(z)
        z = z.view(B, T, P, D).unsqueeze(3).expand(-1, -1, -1, N, -1)
        x = x + z
        # flatten back [B, T, PN, D]
        return x.view(B, T, PN, D)



class TaPformer(BaseModel):
    def __init__(self, ori_parts_idx, reo_parts_idx, reo_all_idx,
                 mask_tensor, supports, patch_adj_matrices, model_args, **args):
        super(TaPformer, self).__init__(**args)
        # hyperparameters
        self.tem_patchsize = model_args['tem_patchsize']
        self.tem_patchnum  = model_args['tem_patchnum']
        self.his_len       = model_args['his_len']
        self.node_num      = model_args['node_num']
        self.tod           = model_args['tod']
        self.dow           = model_args['dow']
        self.layers        = model_args['layers']
        self.input_dims    = model_args['input_dims']
        self.node_dims     = model_args['node_dims']
        self.tod_dims      = model_args['tod_dims']
        self.dow_dims      = model_args['dow_dims']
        # indices & buffers
        self.ori_parts_idx = ori_parts_idx
        self.reo_parts_idx = reo_parts_idx
        self.reo_all_idx   = reo_all_idx
        self.supports = supports
        self.register_buffer('mask', mask_tensor)  # [P, N]
        patch_adj = torch.tensor(patch_adj_matrices, dtype=torch.float32)
        self.register_buffer('patch_adj', patch_adj)  # [P, N, N]
        P = len(self.ori_parts_idx)
        N = self.mask.size(1)
        # embeddings
        self.input_st_fc = nn.Conv2d(3, self.input_dims,
                                     kernel_size=(1, self.tem_patchsize),
                                     stride=(1, self.tem_patchsize))
        self.long_input_linear = nn.Linear(self.his_len, self.input_dims)
        self.node_emb = nn.Parameter(torch.randn(self.node_num, self.node_dims))
        self.time_in_day_emb = nn.Parameter(torch.empty(self.tod, self.tod_dims))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(torch.empty(self.dow, self.dow_dims))
        nn.init.xavier_uniform_(self.day_in_week_emb)
        # model layers
        D = self.input_dims + self.node_dims + self.tod_dims + self.dow_dims
        # D = self.input_dims * 2 + self.node_dims + self.tod_dims + self.dow_dims
        self.layers = nn.ModuleList([
            WindowAttBlock(D, num_heads=1, P=P, N=N, mlp_ratio=1.0)
            for _ in range(self.layers)
        ])
        self.regressor = nn.Conv2d(in_channels=D,
                                   out_channels=self.tem_patchsize * self.tem_patchnum,
                                   kernel_size=(1,1))

    def embedding(self, x, his=None):
        B, T, N0, _ = x.shape
        inp = self.input_st_fc(x.transpose(1,3)).transpose(1,3)
        if his is not None:
            long_inp = self.long_input_linear(his.unsqueeze(-1).transpose(1,3))
        # time embeddings
        td = (x[...,1] * self.tod).long()
        td = self.time_in_day_emb[(td[:, -1, :]).type(torch.LongTensor)].unsqueeze(-1).permute(0, 3, 1, 2)
        dw = x[..., 2] * self.dow
        dw = self.day_in_week_emb[(dw[:, -1, :]).type(torch.LongTensor)].unsqueeze(-1).permute(0, 3, 1, 2)
        nb  = self.node_emb.unsqueeze(0).unsqueeze(0).expand(B, inp.size(1), -1, -1)
        if his is not None:
            return torch.cat([inp, long_inp, td, dw, nb], dim=-1)
        else:
            return torch.cat([inp, td, dw, nb], dim=-1)

    def forward(self, x, label=None, his=None):
        B, T, N0, _ = x.shape
        # emb = self.embedding(x, his[..., 0])  # [B, T', N0, D]
        emb = self.embedding(x)  # [B, T', N0, D]
        rex = emb[:,:, self.reo_all_idx, :]
        # transformer blocks
        for blk in self.layers:
            rex = blk(rex, self.mask, self.patch_adj)
        # restore original order
        D = rex.size(-1)
        out = torch.zeros(B, rex.size(1), N0, D, device=rex.device)
        for ori, reo in zip(self.ori_parts_idx, self.reo_parts_idx):
            L = len(ori)
            out[:,:, ori, :] = rex[:,:, reo[:L], :]
        # regression
        y = self.regressor(out.permute(0,3,2,1))  # [B, H, N0, 1]
        return y
