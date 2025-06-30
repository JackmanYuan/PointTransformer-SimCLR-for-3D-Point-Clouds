# PointTransformer-SimCLR for 3D Point Clouds
We train a SimCLR-style contrastive model for 3D point clouds, using a Point Transformer–based encoder to capture both local and global geometry. This enables robust, self-supervised feature learning from raw RGB-D scans.

---
## 🛠️ Environment Setup

We recommend using **conda** with **Python 3.10** for compatibility with `open3d` and `torch`.

### 1. Create and Activate Conda Environment

```bash
conda create -n simclr3d python=3.10 -y
conda activate simclr3d
```

### 2. Install Dependencies

First, ensure you have your `requirements.txt` ready:

```
numpy==1.23.5
open3d==0.19.0
Pillow==11.2.1
scikit_learn==1.7.0
scipy==1.16.0
torch==2.7.1
torch_cluster==1.6.3+pt27cu126
tqdm==4.67.1
```

Install with:

```bash
pip install -r requirements.txt
```

> ⚠️ Note: If `torch_cluster` fails to install via pip due to PyTorch or CUDA version mismatch, please install it manually using the instructions from the [PyG website](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-via-pip).

Example manual install (for CUDA 12.1):

```bash
pip install torch_cluster -f https://data.pyg.org/whl/torch-2.7.1+cu121.html
```

---
## 📦 Dataset Preparation

This project uses the [SUNRGBD](https://rgbd.cs.princeton.edu/) dataset for point cloud feature learning and matching.

### 1. Download SUNRGBD Dataset

You can download the complete SUNRGBD dataset as a ZIP file:

- **SUNRGBD.zip** :  
  https://rgbd.cs.princeton.edu/data/SUNRGBD.zip

After downloading:

```bash
unzip SUNRGBD.zip -d data/
```

This will extract into:

```
├── kv1
│   ├── b3dodata
│   └── NYUdata
├── kv2
│   ├── align_kv2
│   └── kinect2data
├── realsense
│   ├── lg
│   ├── sa
│   ├── sh
│   └── shr
└── xtion
    ├── sun3ddata
    └── xtion_align_data
```

We can start with testing the **aligned KV2 dataset** found in:

```
data/SUNRGBD/kv2/align_kv2/
```

---

### 2. Convert to PLY and flatten the RGB image and PLY point cloud into two separate folders

Use the script below to convert each RGB-D frame into a `.ply` point cloud and save both the `.ply` and `.jpg` into flat folders:

```bash
python create_SUNRGBD_ply.py \
  --input_root data/SUNRGBD/kv2/align_kv2 \
  --output_image_dir data/rgb \
  --output_ply_dir data/ply
```

This will generate:

```
data/rgb/scene0000_00_img_0000.jpg
data/ply/scene0000_00_img_0000.ply
...
```

Each `.jpg` and `.ply` pair shares the same name prefix: `sceneName_imgName`.

---

## 🏋️‍♂️ Training SimCLR on SUNRGBD Point Clouds

Once your environment and dataset are ready, you can start training the SimCLR model.

### 1. Run Training

This project uses a modified SimCLR pipeline on point clouds, with curvature-weighted contrastive loss and hybrid feature pooling.

Simply run:

```bash
python point_transformer_simclr_v2_curve.py
```

Make sure the following are set inside the script:

- `data/ply` contains the flattened `.ply` files generated from SUNRGBD.
- The output model will be saved as:

```
simclr_encoder_curvature_weighted.pth
```

---

### 2. Training Details

- **Model**: Point Transformer encoder + projection head
- **Loss**: NT-Xent contrastive loss with optional curvature weighting
- **Augmentations**: Random crop, jittering, rotation
- **Feature**: Hybrid global pooling (mean + max)
- **Device**: GPU (if available)
- **Epochs**: 100
- **Batch size**: 32
- **Points per cloud**: Defined in `cfg.num_point` (default: 2048)

---
## 🔍 Visualize Encoder Output

You can **visualize the learned features** and their consistency across augmented views using:

```bash
visualize_encoder.ipynb
```

This notebook performs:
- Loading a single `.ply` point cloud from your dataset
- Creating one augmented version of it (as done during training)
- Passing both through your trained encoder
- Visualizing the feature descriptors in 3D
- Drawing **matching lines** between corresponding points in the two views

![Screenshot 2025-07-01 at 1 03 34 AM](https://github.com/user-attachments/assets/9eb9dfb0-5d74-46f8-9937-7677bcdd2adb)


> 🎯 This helps qualitatively assess how well the encoder produces consistent features across augmented point clouds.

---

## 🔍 Model Evaluation and Comparison

To quantitatively compare different encoders, we use **matching quality** as the evaluation metric.

This script evaluates how well two point cloud encoders match augmented views by:
- Extracting features
- Finding soft correspondences with a matching head
- Estimating a rigid transformation (rotation + translation)
- Computing RMSE between aligned source and target points

---

### ✅ Run Model Comparison

You can use the following script to evaluate and compare two trained encoders:

```bash
python model_compare_by_matching.py
```

And it will:
- Loads two pretrained encoders 
- Uses the matching head to compute soft correspondences
- Estimates transformation between the two views
- Calculates RMSE as the alignment error for each sample

Output:

```
✅ Evaluation Summary:
Encoder 1 - Mean RMSE: 0.6833, Std: 0.2890
Encoder 2 - Mean RMSE: 0.6243, Std: 0.2955
```

You can update paths and filenames in the script to compare any pair of trained models.

---

## 📚 Acknowledgments

This project builds upon the following works and resources:

- **[Point Transformer (Point-Transformers)](https://github.com/qq456cvb/Point-Transformers)**  
  Wu, Z., Xu, S., & Zhou, M. (2021). *Point Transformer*. NeurIPS 2021.  
  We adapt the encoder architecture from this repository.

- **[SimCLR](https://arxiv.org/abs/2002.05709)**  
  Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). *A Simple Framework for Contrastive Learning of Visual Representations*. ICML 2020.  
  Our contrastive training pipeline is inspired by this work.

- **[SUNRGBD Dataset](https://rgbd.cs.princeton.edu/)**  
  Song, S., Lichtenberg, S. P., & Xiao, J. (2015). *SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite*.  
  We use this dataset for generating point clouds and training unsupervised representations.

---

## 📝 License

This project is released under the **MIT License**.

Please also check and follow the licenses of the above referenced works if you use or modify components from their original repositories.
