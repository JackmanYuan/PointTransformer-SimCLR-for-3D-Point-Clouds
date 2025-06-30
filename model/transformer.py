from utils.pointnet_util import index_points, square_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        
    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)
        
        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f
        
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res, attn

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, relative_pos):
        return self.mlp(relative_pos)

class MultiHeadTransformerBlock(nn.Module):
    def __init__(self, dim, dim_out, k=16, num_heads=4):
        super().__init__()
        self.k = k
        self.num_heads = num_heads
        self.dim = dim
        self.dim_out = dim_out
        self.head_dim = dim_out // num_heads

        assert self.head_dim * num_heads == dim_out, "dim_out must be divisible by num_heads"

        self.to_q = nn.Linear(dim, dim_out)
        self.to_k = nn.Linear(dim, dim_out)
        self.to_v = nn.Linear(dim, dim_out)

        self.pos_enc = PositionalEncoding(self.head_dim)

        self.fc = nn.Sequential(
            nn.Linear(dim_out, dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, dim_out)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim_out)

    def forward(self, xyz, features):
        B, N, _ = xyz.shape

        # KNN: find neighbors
        dist = torch.cdist(xyz, xyz)  # [B, N, N]
        knn_idx = dist.topk(self.k, largest=False)[1]  # [B, N, k]

        # Gather neighbors
        idx_base = torch.arange(0, B, device=xyz.device).view(-1, 1, 1) * N
        knn_idx = knn_idx + idx_base
        knn_idx = knn_idx.view(-1)

        xyz_flat = xyz.view(B * N, -1)
        neighbor_xyz = xyz_flat[knn_idx].view(B, N, self.k, 3)
        relative_pos = neighbor_xyz - xyz.unsqueeze(2)  # [B, N, k, 3]
        pos_enc = self.pos_enc(relative_pos)  # [B, N, k, head_dim]

        feat_flat = features.view(B * N, -1)
        q = self.to_q(features).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, N, d_h]
        k = self.to_k(feat_flat[knn_idx].view(B, N, self.k, -1))  # [B, N, k, dim_out]
        k = k.view(B, N, self.k, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # [B, h, N, k, d_h]

        v = self.to_v(feat_flat[knn_idx].view(B, N, self.k, -1))
        v = v.view(B, N, self.k, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # [B, h, N, k, d_h]

        # Attention with positional bias
        q = q.unsqueeze(3)  # [B, h, N, 1, d_h]
        pos_enc = pos_enc.view(B, N, self.k, 1, self.head_dim).permute(0, 3, 1, 2, 4)  # [B, h, N, k, d_h]

        attn = (q * (k + pos_enc)).sum(-1) / (self.head_dim ** 0.5)  # [B, h, N, k]
        attn = F.softmax(attn, dim=-1)

        out = (attn.unsqueeze(-1) * (v + pos_enc)).sum(3)  # [B, h, N, d_h]
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.dim_out)  # [B, N, dim_out]

        out = self.norm2(self.fc(out)) + self.to_q(self.norm1(features))  # Residual
        return out, attn
    