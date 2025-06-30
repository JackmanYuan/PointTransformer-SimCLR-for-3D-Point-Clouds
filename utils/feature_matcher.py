import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.model import Backbone
from model.matcher import MatchingHead
from utils.loss import matching_loss
from data_loader import MatchingHeadDataset
from utils.config import get_default_cfg

# --------- Config & Setup ---------
cfg = get_default_cfg()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder_ckpt_path = "simclr_encoder.pth"
save_path = "matching_head.pth"

# --------- Load Pretrained Encoder ---------
encoder = Backbone(cfg).to(device)
ckpt = torch.load(encoder_ckpt_path, map_location=device)
encoder.load_state_dict(ckpt["encoder"])
encoder.eval()  # freeze encoder
for param in encoder.parameters():
    param.requires_grad = False

# --------- Matching Head ---------
matching_head = MatchingHead().to(device)
optimizer = torch.optim.Adam(matching_head.parameters(), lr=1e-3)

# --------- Dataset & Dataloader ---------
train_dataset = MatchingHeadDataset(
    root_folder="/home/jack/Projects/point_transformer_unsup/data/SUNRGBD/data_autoencoder",
    num_points=cfg.num_point,
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

# --------- Training Loop ---------
best_loss = float("inf")
for epoch in range(1, 101):
    matching_head.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc=f"[Epoch {epoch}] Matching Head Training", leave=False)
    for view1, view2, meta in pbar:
        view1, view2 = view1.to(device), view2.to(device)
        xyz1, xyz2 = view1[:, :, :3], view2[:, :, :3]
        R1 = torch.stack([torch.tensor(r) for r in meta["R1"]], dim=0).to(device)  # [B, 3, 3]
        R2 = torch.stack([torch.tensor(r) for r in meta["R2"]], dim=0).to(device)

        # Encode features without gradient
        with torch.no_grad():
            _, td1 = encoder(view1)
            _, td2 = encoder(view2)
            F1 = td1[-1][1]  # [B, N, D]
            F2 = td2[-1][1]

        # Compute soft correspondence + loss
        soft_corr = matching_head(F1, F2)
        loss = matching_loss(soft_corr, xyz1, xyz2, R1, R2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch}] Avg Matching Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            "epoch": epoch,
            "matching_head": matching_head.state_dict(),
            "loss": best_loss
        }, save_path)
        print(f"✓ New best model saved at epoch {epoch} with loss {best_loss:.4f}")