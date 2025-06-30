import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.model import Backbone
from model.matcher import MatchingHead
from utils.config import get_default_cfg
from utils.data_loader import SimCLREvalDataset
from scipy.spatial.transform import Rotation as R

# ----------------------------
# Config & Settings
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layer_id = 0
top_k = 100
ply_folder = "data/ply" 
max_files = 1000
batch_size = 1  

# ----------------------------
# Load Dataset
# ----------------------------
dataset = SimCLREvalDataset(ply_folder=ply_folder, num_points=1024, normalize=True, max_files=max_files)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# ----------------------------
# Load Models
# ----------------------------
cfg = get_default_cfg()

encoder1 = Backbone(cfg).to(device)
ckpt1 = torch.load("model_archive/simclr_encoder_0630.pth", map_location=device) # Use your owwn path
encoder1.load_state_dict(ckpt1["encoder"])
encoder1.eval()

encoder2 = Backbone(cfg).to(device)
ckpt2 = torch.load("model_archive/simclr_encoder_curvature_weighted_0630.pth", map_location=device) # Use your owwn path
encoder2.load_state_dict(ckpt2["encoder"])
encoder2.eval()

matching_head = MatchingHead().to(device)
matching_head.eval()

# ----------------------------
# Evaluation Function
# ----------------------------


def estimate_rigid_transform(src, tgt):
    """
    Estimate rotation + translation (rigid) using the Kabsch algorithm.
    """
    assert src.shape == tgt.shape
    mu_src = src.mean(0)
    mu_tgt = tgt.mean(0)
    src_centered = src - mu_src
    tgt_centered = tgt - mu_tgt
    H = src_centered.T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)
    R_est = Vt.T @ U.T
    if np.linalg.det(R_est) < 0:
        Vt[-1, :] *= -1
        R_est = Vt.T @ U.T
    t_est = mu_tgt - R_est @ mu_src
    return R_est, t_est


def evaluate(encoder, label):
    scores = []
    for pc1, pc2, _ in tqdm(loader, desc=f"Evaluating {label}"):
        pc1, pc2 = pc1.to(device), pc2.to(device)

        with torch.no_grad():
            _, f1_feats = encoder(pc1)
            _, f2_feats = encoder(pc2)

        f1 = f1_feats[layer_id][1].squeeze(0).cpu().numpy()
        f2 = f2_feats[layer_id][1].squeeze(0).cpu().numpy()
        xyz1 = pc1[0, :, :3].cpu().numpy()
        xyz2 = pc2[0, :, :3].cpu().numpy()

        # Compute soft correspondence
        f1_tensor = torch.from_numpy(f1).unsqueeze(0).to(device)
        f2_tensor = torch.from_numpy(f2).unsqueeze(0).to(device)
        with torch.no_grad():
            soft_corr = matching_head(f1_tensor, f2_tensor)[0].squeeze(0)
        top_idx = torch.argmax(soft_corr, dim=1).cpu().numpy()

        # Top-K feature filtering
        significance = np.linalg.norm(f1, axis=1)
        topk_idx = np.argsort(significance)[-top_k:]
        src = xyz1[topk_idx]
        tgt = xyz2[top_idx[topk_idx]]

        # === Estimate rigid transformation (no scaling) ===
        R_est, t_est = estimate_rigid_transform(src, tgt)
        src_aligned = (R_est @ src.T).T + t_est  # Apply rotation and translation
        
        # === RMSE after alignment ===
        rmse = np.sqrt(np.mean(np.sum((src_aligned - tgt) ** 2, axis=1)))
        scores.append(rmse)

    return scores

# ----------------------------
# Run Evaluation
# ----------------------------
scores1 = evaluate(encoder1, "Encoder 1")
scores2 = evaluate(encoder2, "Encoder 2")

print("\n✅ Evaluation Summary:")
print(f"Encoder 1 - Mean RMSE: {np.mean(scores1):.4f}, Std: {np.std(scores1):.4f}")
print(f"Encoder 2 - Mean RMSE: {np.mean(scores2):.4f}, Std: {np.std(scores2):.4f}")