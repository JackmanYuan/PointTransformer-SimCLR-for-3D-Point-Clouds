import os
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset

# class SimCLRPointCloudDataset(Dataset):
#     def __init__(self, root_folder, num_points=2048, normalize=True):
#         self.root_folder = root_folder
#         self.num_points = num_points
#         self.normalize = normalize
#         self.files = []

#         for scene_folder in sorted(os.listdir(root_folder)):
#             for subfolder in ["vggt", "rgbd"]:
#                 scene_path = os.path.join(root_folder, scene_folder, subfolder)
#                 if os.path.exists(scene_path):
#                     for file in sorted(os.listdir(scene_path)):
#                         if file.endswith(".ply"):
#                             self.files.append(os.path.join(scene_path, file))

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         pcd = o3d.io.read_point_cloud(self.files[idx])
#         points = np.asarray(pcd.points, dtype=np.float32)
#         colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else np.zeros_like(points)

#         if self.normalize:
#             centroid = np.mean(points, axis=0)
#             points -= centroid
#             scale = np.max(np.linalg.norm(points, axis=1))
#             points /= scale

#         # Generate two views
#         view1 = self._augment(points, colors)
#         view2 = self._augment(points, colors)

#         return view1, view2, self.files[idx]  # Added file path for consistency

#     def _augment(self, points, colors):
#         pc = points.copy()

#         # === Step 1: Safer Axis-Aligned Cropping with Fallback ===
#         crop_axis = np.random.choice(3)
#         crop_ratio = 0.7  # Keep 70% of points
#         min_val = np.min(pc[:, crop_axis])
#         max_val = np.max(pc[:, crop_axis])
#         crop_threshold = np.random.uniform(min_val, max_val)
#         mask = pc[:, crop_axis] < crop_threshold

#         # Fallback: select top-N if too few remain
#         if np.sum(mask) < 512:  # Minimum point fallback
#             sorted_idx = np.argsort(pc[:, crop_axis])
#             mask = np.zeros(len(pc), dtype=bool)
#             mask[sorted_idx[:int(crop_ratio * len(pc))]] = True

#         pc = pc[mask]
#         colors = colors[mask]

#         # === Step 2: Sampling to num_points ===
#         if len(pc) >= self.num_points:
#             idxs = np.random.choice(len(pc), self.num_points, replace=False)
#         else:
#             idxs = np.random.choice(len(pc), self.num_points, replace=True)
#         pc = pc[idxs]
#         colors = colors[idxs]

#         # === Step 3: Jitter ===
#         pc += np.random.normal(0, 0.01, size=pc.shape)

#         # === Step 4: Rotation (XYZ) ===
#         angles = np.random.uniform(0, 2 * np.pi, size=3)
#         Rx = np.array([[1, 0, 0],
#                     [0, np.cos(angles[0]), -np.sin(angles[0])],
#                     [0, np.sin(angles[0]),  np.cos(angles[0])]])
#         Ry = np.array([[ np.cos(angles[1]), 0, np.sin(angles[1])],
#                     [0, 1, 0],
#                     [-np.sin(angles[1]), 0, np.cos(angles[1])]])
#         Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
#                     [np.sin(angles[2]),  np.cos(angles[2]), 0],
#                     [0, 0, 1]])
#         pc = pc @ (Rx @ Ry @ Rz).T

#         # === Step 5: Combine and return ===
#         feats = np.concatenate([pc, colors], axis=1)
#         return torch.from_numpy(feats).float()
    

class SimCLRPointCloudDataset(Dataset):
    def __init__(self, ply_folder, num_points=2048, normalize=True):
        self.ply_folder = ply_folder
        self.num_points = num_points
        self.normalize = normalize
        self.files = sorted([os.path.join(ply_folder, f)
                             for f in os.listdir(ply_folder) if f.endswith('.ply')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pcd = o3d.io.read_point_cloud(self.files[idx])
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else np.zeros_like(points)

        if self.normalize:
            centroid = np.mean(points, axis=0)
            points -= centroid
            scale = np.max(np.linalg.norm(points, axis=1))
            points /= scale

        view1 = self._augment(points, colors)
        view2 = self._augment(points, colors)

        return view1, view2, self.files[idx]

    def _augment(self, points, colors):
        pc = points.copy()

        crop_axis = np.random.choice(3)
        crop_ratio = 0.7
        min_val = np.min(pc[:, crop_axis])
        max_val = np.max(pc[:, crop_axis])
        crop_threshold = np.random.uniform(min_val, max_val)
        mask = pc[:, crop_axis] < crop_threshold

        if np.sum(mask) < 512:
            sorted_idx = np.argsort(pc[:, crop_axis])
            mask = np.zeros(len(pc), dtype=bool)
            mask[sorted_idx[:int(crop_ratio * len(pc))]] = True

        pc = pc[mask]
        colors = colors[mask]

        idxs = np.random.choice(len(pc), self.num_points, replace=len(pc) < self.num_points)
        pc = pc[idxs]
        colors = colors[idxs]

        pc += np.random.normal(0, 0.01, size=pc.shape)

        angles = np.random.uniform(0, 2 * np.pi, size=3)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]),  np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]),  np.cos(angles[2]), 0],
                       [0, 0, 1]])
        pc = pc @ (Rx @ Ry @ Rz).T

        feats = np.concatenate([pc, colors], axis=1)
        return torch.from_numpy(feats).float()

class SimCLRPointCloudDatasetV2(Dataset):
    def __init__(self, ply_folder, num_points=2048, normalize=True):
        self.ply_folder = ply_folder
        self.num_points = num_points
        self.normalize = normalize
        self.files = sorted([os.path.join(ply_folder, f)
                             for f in os.listdir(ply_folder) if f.endswith('.ply')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pcd = o3d.io.read_point_cloud(self.files[idx])
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else np.zeros_like(points)

        view1 = self._augment_then_normalize(points, colors)
        view2 = self._augment_then_normalize(points, colors)

        return view1, view2, self.files[idx]

    def _augment_then_normalize(self, points, colors):
        pc = points.copy()
        c = colors.copy()

        # === Augmentation ===
        # Crop along random axis
        crop_axis = np.random.choice(3)
        crop_ratio = 0.7
        min_val = np.min(pc[:, crop_axis])
        max_val = np.max(pc[:, crop_axis])
        crop_threshold = np.random.uniform(min_val, max_val)
        mask = pc[:, crop_axis] < crop_threshold

        # Fallback if too few remain
        if np.sum(mask) < 512:
            sorted_idx = np.argsort(pc[:, crop_axis])
            mask = np.zeros(len(pc), dtype=bool)
            mask[sorted_idx[:int(crop_ratio * len(pc))]] = True

        pc = pc[mask]
        c = c[mask]

        # Random uniform scaling
        scale_factor = np.random.uniform(0.8, 1.2)
        pc *= scale_factor

        # Sampling
        if len(pc) >= self.num_points:
            idxs = np.random.choice(len(pc), self.num_points, replace=False)
        else:
            idxs = np.random.choice(len(pc), self.num_points, replace=True)
        pc = pc[idxs]
        c = c[idxs]

        # Jitter
        pc += np.random.normal(0, 0.01, size=pc.shape)

        # Random rotation
        angles = np.random.uniform(0, 2 * np.pi, size=3)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]),  np.cos(angles[0])]])
        Ry = np.array([[ np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]),  np.cos(angles[2]), 0],
                       [0, 0, 1]])
        pc = pc @ (Rx @ Ry @ Rz).T

        # === Normalize ===
        centroid = np.mean(pc, axis=0)
        pc -= centroid
        scale = np.max(np.linalg.norm(pc, axis=1))
        pc /= scale

        # === Combine ===
        feats = np.concatenate([pc, c], axis=1)
        return torch.from_numpy(feats).float()


class SimCLREvalDataset(Dataset):
    def __init__(self, ply_folder, num_points=2048, normalize=True, max_files=300):
        self.ply_folder = ply_folder
        self.num_points = num_points
        self.normalize = normalize

        all_files = sorted([os.path.join(ply_folder, f)
                            for f in os.listdir(ply_folder) if f.endswith('.ply')])
        self.files = all_files[:max_files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pcd = o3d.io.read_point_cloud(self.files[idx])
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else np.zeros_like(points)

        if self.normalize:
            centroid = np.mean(points, axis=0)
            points -= centroid
            scale = np.max(np.linalg.norm(points, axis=1))
            points /= scale

        view1 = self._augment(points, colors)
        view2 = self._augment(points, colors)

        return view1, view2, self.files[idx]

    def _augment(self, points, colors):
        pc = points.copy()

        crop_axis = np.random.choice(3)
        crop_ratio = 0.7
        min_val = np.min(pc[:, crop_axis])
        max_val = np.max(pc[:, crop_axis])
        crop_threshold = np.random.uniform(min_val, max_val)
        mask = pc[:, crop_axis] < crop_threshold

        if np.sum(mask) < 512:
            sorted_idx = np.argsort(pc[:, crop_axis])
            mask = np.zeros(len(pc), dtype=bool)
            mask[sorted_idx[:int(crop_ratio * len(pc))]] = True

        pc = pc[mask]
        colors = colors[mask]

        idxs = np.random.choice(len(pc), self.num_points, replace=len(pc) < self.num_points)
        pc = pc[idxs]
        colors = colors[idxs]

        pc += np.random.normal(0, 0.01, size=pc.shape)

        angles = np.random.uniform(0, 2 * np.pi, size=3)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]),  np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]),  np.cos(angles[2]), 0],
                       [0, 0, 1]])
        pc = pc @ (Rx @ Ry @ Rz).T

        feats = np.concatenate([pc, colors], axis=1)
        return torch.from_numpy(feats).float()

class MatchingHeadDataset(Dataset):
    def __init__(self, ply_folder, num_points=2048, normalize=True):

        self.ply_folder = ply_folder
        self.num_points = num_points
        self.normalize = normalize
        self.files = sorted([os.path.join(ply_folder, f)
                             for f in os.listdir(ply_folder) if f.endswith('.ply')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load point cloud
        pcd = o3d.io.read_point_cloud(self.files[idx])
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else np.zeros_like(points)

        # Normalize
        if self.normalize:
            centroid = np.mean(points, axis=0)
            points -= centroid
            scale = np.max(np.linalg.norm(points, axis=1))
            points /= scale
        else:
            centroid = np.zeros(3, dtype=np.float32)
            scale = 1.0

        # Augment twice with transformation tracking
        view1, R1, mask1 = self._augment(points, colors)
        view2, R2, mask2 = self._augment(points, colors)

        meta = {
            "file": self.files[idx],
            "centroid": centroid.astype(np.float32),
            "scale": float(scale),
            "R1": R1.astype(np.float32),
            "R2": R2.astype(np.float32),
            "mask1": mask1.astype(np.bool_),
            "mask2": mask2.astype(np.bool_),
        }

        return view1, view2, meta

    def _augment(self, points, colors):
        pc = points.copy()

        # === Step 1: Cropping ===
        crop_axis = np.random.choice(3)
        crop_ratio = 0.7
        min_val = np.min(pc[:, crop_axis])
        max_val = np.max(pc[:, crop_axis])
        crop_threshold = np.random.uniform(min_val, max_val)
        mask = pc[:, crop_axis] < crop_threshold

        if np.sum(mask) < 512:
            sorted_idx = np.argsort(pc[:, crop_axis])
            mask = np.zeros(len(pc), dtype=bool)
            mask[sorted_idx[:int(crop_ratio * len(pc))]] = True

        pc = pc[mask]
        colors = colors[mask]

        # === Step 2: Sampling ===
        if len(pc) >= self.num_points:
            idxs = np.random.choice(len(pc), self.num_points, replace=False)
        else:
            idxs = np.random.choice(len(pc), self.num_points, replace=True)

        pc = pc[idxs]
        colors = colors[idxs]
        mask_out = np.ones(self.num_points, dtype=bool)

        # === Step 3: Jitter ===
        pc += np.random.normal(0, 0.01, size=pc.shape)

        # === Step 4: Rotation ===
        angles = np.random.uniform(0, 2 * np.pi, size=3)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]),  np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]),  np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = Rx @ Ry @ Rz
        pc = pc @ R.T  # Apply rotation

        # === Step 5: Concatenate point + color ===
        feats = np.concatenate([pc, colors], axis=1)  # [N, 6]
        return torch.from_numpy(feats).float(), R, mask_out
