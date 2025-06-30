import os
import shutil
import argparse
from tqdm import tqdm
import open3d as o3d
from PIL import Image

def load_intrinsics(intrinsics_path, image_path):
    img = Image.open(image_path)
    width, height = img.size

    with open(intrinsics_path, 'r') as f:
        vals = list(map(float, f.read().strip().split()))
    fx, fy, cx, cy = vals[0], vals[4], vals[2], vals[5]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

def process_one_scene(scene_path, output_image_dir, output_ply_dir):
    image_dir = os.path.join(scene_path, "image")
    depth_dir = os.path.join(scene_path, "depth")
    intr_path = os.path.join(scene_path, "intrinsics.txt")

    if not (os.path.exists(image_dir) and os.path.exists(depth_dir) and os.path.exists(intr_path)):
        print(f"[!] Skipping {scene_path}: missing folders or intrinsics.txt")
        return 0

    scene_name = os.path.basename(scene_path)
    processed_count = 0

    for fname in sorted(os.listdir(image_dir)):
        if not fname.endswith(".jpg"):
            continue

        rgb_path = os.path.join(image_dir, fname)
        depth_path = os.path.join(depth_dir, fname.replace(".jpg", ".png"))

        if not os.path.exists(depth_path):
            print(f"[!] Missing depth for {fname}, skipping.")
            continue

        try:
            intrinsics = load_intrinsics(intr_path, rgb_path)
        except Exception as e:
            print(f"[!] Error loading intrinsics for {scene_path}: {e}")
            continue

        base_name = f"{scene_name}_{fname}"
        ply_name = base_name.replace(".jpg", ".ply")

        shutil.copy(rgb_path, os.path.join(output_image_dir, base_name))

        color = o3d.io.read_image(rgb_path)
        depth = o3d.io.read_image(depth_path)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth,
            convert_rgb_to_intensity=False,
            depth_scale=1000.0,
            depth_trunc=50.0
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)

        o3d.io.write_point_cloud(os.path.join(output_ply_dir, ply_name), pcd)
        processed_count += 1

    return processed_count

def batch_process_flat(input_root, output_image_dir, output_ply_dir):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_ply_dir, exist_ok=True)

    scene_dirs = [os.path.join(input_root, d) for d in sorted(os.listdir(input_root)) if os.path.isdir(os.path.join(input_root, d))]

    total = 0
    for scene_path in tqdm(scene_dirs, desc="Processing scenes"):
        total += process_one_scene(scene_path, output_image_dir, output_ply_dir)

    print(f"\n✅ Done! Total frames processed: {total}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SUNRGBD RGBD images to PLY point clouds in flat format.")
    parser.add_argument("--input_root", type=str, default="data/SUNRGBD/kv1/NYUdata", help="Root folder of SUNRGBD raw data")
    parser.add_argument("--output_image_dir", type=str, default="data/rgb", help="Folder to save flattened RGB images")
    parser.add_argument("--output_ply_dir", type=str, default="data/ply", help="Folder to save flattened PLY files")

    args = parser.parse_args()
    batch_process_flat(args.input_root, args.output_image_dir, args.output_ply_dir)