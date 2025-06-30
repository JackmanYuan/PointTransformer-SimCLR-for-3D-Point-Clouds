import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchingHead(nn.Module):
    def __init__(self, temperature=0.07, k=32):
        super().__init__()
        self.temperature = temperature
        self.k = k

    def forward(self, F1, F2):
        """
        F1: [B, N, D] - features from view 1
        F2: [B, N, D] - features from view 2
        Returns:
            soft_corr: [B, N, N] - correspondence matrix (softmax over top-k in F2)
        """
        B, N, D = F1.shape

        # Normalize features
        F1 = F.normalize(F1, dim=-1)
        F2 = F.normalize(F2, dim=-1)

        # Similarity matrix: [B, N, N]
        sim = torch.bmm(F1, F2.transpose(1, 2)) / self.temperature  # [B, N, N]

        # Top-k indices and scores: [B, N, k]
        topk_scores, topk_idx = torch.topk(sim, self.k, dim=-1)  # [B, N, k]

        # Apply softmax to top-k scores
        topk_weights = F.softmax(topk_scores, dim=-1)  # [B, N, k]

        # Create zero-filled matrix and scatter softmax scores into it
        soft_corr = torch.zeros_like(sim)  # [B, N, N]
        soft_corr.scatter_(dim=-1, index=topk_idx, src=topk_weights)
        
        return soft_corr, topk_idx