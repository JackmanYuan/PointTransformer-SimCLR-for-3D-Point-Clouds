import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.model import Backbone
from model.projection import ProjectionHead
from utils.data_loader import SimCLRPointCloudDataset
from utils.loss import nt_xent_loss_with_weights
from utils.config import get_default_cfg
from utils.estimate_curvature import estimate_curvature 

# --------- Config & Setup ---------
cfg = get_default_cfg()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = "simclr_encoder_curvature_weighted.pth"

# --------- Model (Encoder + Projection) ---------
encoder = Backbone(cfg).to(device)
projector = ProjectionHead(input_dim=512).to(device)  # 256 mean + 256 max
params = list(encoder.parameters()) + list(projector.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)

# --------- Dataset & Dataloader ---------
train_dataset = SimCLRPointCloudDataset(
    ply_folder="data/ply",
    num_points=cfg.num_point,
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

# --------- Training Loop ---------
best_loss = float("inf")
for epoch in range(1, 101):
    encoder.train()
    projector.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc=f"[Epoch {epoch}] SimCLR Training", leave=False)
    for view1, view2, _ in pbar:
        view1 = view1.to(device)  # [B, N, C]
        view2 = view2.to(device)

        # Estimate curvature from XYZ coordinates
        xyz1 = view1[:, :, :3]
        xyz2 = view2[:, :, :3]
        curvature1 = estimate_curvature(xyz1)  # [B, N]
        curvature2 = estimate_curvature(xyz2)
        weights = (curvature1 + curvature2) / 2.0
        weights = weights / (weights.mean() + 1e-6)  # [B, N] normalized

        # Forward pass
        _, td1 = encoder(view1)
        _, td2 = encoder(view2)

        feat1 = td1[-1][1]  # [B, N, D]
        feat2 = td2[-1][1]

        # Hybrid pooling: mean + max
        pooled1 = torch.cat([feat1.mean(dim=1), feat1.max(dim=1)[0]], dim=1)  # [B, 2D]
        pooled2 = torch.cat([feat2.mean(dim=1), feat2.max(dim=1)[0]], dim=1)

        z1 = projector(pooled1)
        z2 = projector(pooled2)

        # Use averaged curvature per sample as weight
        loss = nt_xent_loss_with_weights(z1, z2)  # [B]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch}] Avg Contrastive Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "projector": projector.state_dict(),
            "loss": best_loss
        }, save_path)
        print(f"✓ New best model saved at epoch {epoch} with loss {best_loss:.4f}")