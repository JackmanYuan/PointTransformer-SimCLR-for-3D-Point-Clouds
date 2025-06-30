import torch
import torch.nn.functional as F

def estimate_curvature(xyz, k=20):
    """
    Estimate curvature from point cloud using eigenvalue analysis of local neighborhoods.

    Args:
        xyz: [B, N, 3] point cloud coordinates
        k: number of neighbors for each point

    Returns:
        curvature: [B, N] tensor – estimated curvature values
    """
    from torch_cluster import knn

    B, N, _ = xyz.shape
    xyz_flat = xyz.view(B * N, 3)
    batch = torch.arange(B, device=xyz.device).repeat_interleave(N)

    idx = knn(xyz_flat, xyz_flat, k=k, batch_x=batch, batch_y=batch)  # (2, B*N*k)

    # Gather neighbors and compute PCA-like curvature estimate
    src, dst = idx
    diffs = xyz_flat[dst] - xyz_flat[src]  # [B*N*k, 3]

    # Reshape to [B*N, k, 3]
    diffs = diffs.view(B * N, k, 3)
    cov = torch.matmul(diffs.transpose(1, 2), diffs) / k  # [B*N, 3, 3]

    # Eigenvalues of covariance matrix
    eigvals = torch.linalg.eigvalsh(cov)  # [B*N, 3]
    eigvals = F.relu(eigvals)  # Remove negative small values

    curvature = eigvals[:, 0] / (eigvals.sum(dim=1) + 1e-8)  # Smallest / sum

    return curvature.view(B, N)