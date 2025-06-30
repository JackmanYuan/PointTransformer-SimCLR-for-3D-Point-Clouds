# utils/loss.py
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


def chamfer_loss(pc1, pc2):
    # pc1, pc2: [B, N, 3]
    x, y = pc1, pc2

    x2 = (x ** 2).sum(dim=2, keepdim=True)  # [B, N, 1]
    y2 = (y ** 2).sum(dim=2, keepdim=True).transpose(1, 2)  # [B, 1, N]
    dist = x2 + y2 - 2 * torch.bmm(x, y.transpose(1, 2))  # [B, N, N]

    dist1 = dist.min(dim=2)[0]  # closest in y for each x
    dist2 = dist.min(dim=1)[0]  # closest in x for each y

    return dist1.mean() + dist2.mean()


def contrastive_overlap_loss(xyz_a, feat_a, xyz_b, feat_b, threshold=0.05):
    dist = torch.cdist(xyz_a, xyz_b)  # [B, N_a, N_b]
    min_ab, idx_ab = dist.min(dim=2)
    mask_ab = min_ab < threshold

    matched_feat_b = torch.gather(
        feat_b, 1,
        idx_ab.unsqueeze(-1).expand(-1, -1, feat_b.shape[-1])
    )

    sim = F.cosine_similarity(feat_a, matched_feat_b, dim=-1)
    if mask_ab.sum() == 0:
        return torch.tensor(0.0, device=feat_a.device, requires_grad=True)
    return 1.0 - sim[mask_ab].mean()


def nt_xent_loss(z1, z2, temperature=0.1):
    B, D = z1.shape
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = torch.matmul(z, z.T) / temperature  # [2B, 2B]

    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)

    targets = torch.arange(B, device=z.device)
    positives = torch.cat([targets + B, targets], dim=0)

    loss = F.cross_entropy(sim, positives)
    return loss

def nt_xent_loss_with_weights(z1, z2, curvature=None, temperature=0.1):
    """
    Compute NT-Xent loss with optional curvature-based weighting.

    Args:
        z1, z2: [B, D] tensors – contrastive features
        curvature: [B] or [2B] optional – sample-level weights
        temperature: contrastive temperature
    """
    B, D = z1.shape
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = torch.matmul(z, z.T) / temperature  # [2B, 2B]

    # Mask self-similarity
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)

    # Construct targets (positive pairs)
    targets = torch.arange(B, device=z.device)
    positives = torch.cat([targets + B, targets], dim=0)  # [2B]

    # Cross entropy loss with optional weighting
    if curvature is not None:
        # Normalize and repeat for both views
        weights = curvature / curvature.sum()
        weights = torch.cat([weights, weights], dim=0)  # [2B]
        loss = F.cross_entropy(sim, positives, reduction='none')
        loss = (weights * loss).sum()
    else:
        loss = F.cross_entropy(sim, positives)

    return loss

def point_contrast_loss(anchor_feats, anchor_coords, positive_feats, positive_coords, temperature=0.07):
    """
    Compute contrastive loss between two augmented views of a point cloud.

    Args:
        anchor_feats: (B, N, D)
        anchor_coords: (B, N, 3)
        positive_feats: (B, M, D)
        positive_coords: (B, M, 3)
        temperature: temperature scaling factor

    Returns:
        Scalar contrastive loss
    """
    B, N, D = anchor_feats.shape
    _, M, _ = positive_feats.shape

    # Normalize features
    anchor_feats = F.normalize(anchor_feats, dim=2)    # (B, N, D)
    positive_feats = F.normalize(positive_feats, dim=2)

    # Pairwise Euclidean distance between coords
    dists = torch.cdist(anchor_coords, positive_coords, p=2)  # (B, N, M)
    nn_idx = dists.argmin(dim=2)  # (B, N) → index of closest positive for each anchor

    # Similarity matrix between all anchor-positive pairs
    sim_matrix = torch.bmm(anchor_feats, positive_feats.transpose(1, 2))  # (B, N, M)
    sim_matrix = sim_matrix / temperature

    # Flatten similarity matrix and targets for CE
    sim_matrix_flat = sim_matrix.reshape(B * N, M)     # [B*N, M]
    targets_flat = nn_idx.reshape(-1)                  # [B*N]

    loss = F.cross_entropy(sim_matrix_flat, targets_flat)
    return loss

def matching_loss(soft_corr, xyz1, xyz2, R1, R2, topk_idx, idx1=None):
    # Apply R2 to xyz2: [B, N, 3]
    xyz2_rot = torch.bmm(xyz2, R2.transpose(1, 2))  # [B, N, 3]

    # Apply R1 to xyz1: [B, N, 3] → [B, 32, 3] if idx1 provided
    if idx1 is not None:
        xyz1_rot = torch.stack([
            xyz1[b, idx1[b]] @ R1[b].T for b in range(xyz1.shape[0])
        ])
    else:
        xyz1_rot = torch.bmm(xyz1, R1.transpose(1, 2))  # [B, N, 3]

    # Gather top-k points from xyz2_rot
    B, N, k = topk_idx.shape
    topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    xyz2_topk = torch.gather(xyz2_rot.unsqueeze(1).expand(-1, N, -1, -1), 2, topk_idx_exp)

    # Gather weights
    weights = soft_corr.gather(dim=2, index=topk_idx)  # [B, N, k]
    xyz2_matched = torch.sum(weights.unsqueeze(-1) * xyz2_topk, dim=2)  # [B, N, 3]

    # Ensure xyz1_rot is also [B, N, 3] (same as xyz2_matched)
    if xyz1_rot.shape[1] != xyz2_matched.shape[1]:
        raise ValueError(f"xyz1_rot shape {xyz1_rot.shape} does not match xyz2_matched {xyz2_matched.shape}")

    return F.mse_loss(xyz2_matched, xyz1_rot)