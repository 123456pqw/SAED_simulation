import os
import re
import json
import uuid
import random
import shutil
import argparse
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from tools.Trainer_2 import ModelNetTrainer
from tools.ImgDataset_random_aug import MultiviewImgDataset, SingleImgDataset
from models.MVBCNN import MVBCNN, SVBCNN


torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True


# ----------------------------
# Utils
# ----------------------------
def create_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    else:
        print("WARNING: Summary folder already exists! It will be overwritten!")
        shutil.rmtree(log_dir)
        os.makedirs(log_dir)


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


def mv_collate_fn(batch):
    class_id_h1_list = []
    class_id_h2_list = []
    imgs_list = []
    paths_list = []

    for class_id_h1, class_id_h2, imgs, paths in batch:
        class_id_h1_list.append(class_id_h1)
        class_id_h2_list.append(class_id_h2)
        imgs_list.append(imgs)
        paths_list.append(paths)

    return (
        torch.tensor(class_id_h1_list),
        torch.tensor(class_id_h2_list),
        torch.stack(imgs_list),  # [B, V, C, H, W]
        paths_list,
    )


def load_materials_by_spacegroup(csv_path, root_dir):
    """
    只适配 CSV 列:
    - Material ID
    - Crystal System
    - Space Group Number
    """
    df = pd.read_csv(csv_path)
    required_cols = {"Material ID", "Crystal System", "Space Group Number"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    spacegroup_to_materials = defaultdict(list)
    material_id_dict = {}

    for _, row in df.iterrows():
        mid = str(row["Material ID"]).strip()
        crystal_system = str(row["Crystal System"]).strip()
        if pd.isna(row["Space Group Number"]):
            continue
        sg = int(row["Space Group Number"])

        # 只保留 root_dir 下确实存在数据目录的材料
        mat_dir = os.path.join(root_dir, str(sg), mid)
        if not os.path.isdir(mat_dir):
            continue

        material_id_dict[mid] = crystal_system
        spacegroup_to_materials[sg].append(mid)

    return spacegroup_to_materials, material_id_dict


def split_by_spacegroup(
    spacegroup_to_materials,
    root_dir,
    material_id_dict,
    num_views=2,
    seed=42,
    val_ratio=0.15,
    test_ratio=0.15,
    max_combs_per_material=10,
    oversample_crystals=("cubic", "triclinic"),
    oversample_factor=1,
):
    split_dict = {}
    np.random.seed(seed)
    random.seed(seed)

    for sg, materials in spacegroup_to_materials.items():
        for mid in materials:
            material_dir = os.path.join(root_dir, str(sg), mid)
            if not os.path.isdir(material_dir):
                continue

            views = sorted([f for f in os.listdir(material_dir) if f.endswith(".png")])
            if len(views) < num_views:
                continue

            all_combs = list(combinations(views, num_views))

            crystal_system = material_id_dict.get(mid, None)
            factor = oversample_factor if crystal_system in oversample_crystals else 1

            if max_combs_per_material is not None and len(all_combs) > max_combs_per_material:
                all_combs = random.sample(all_combs, max_combs_per_material)

            all_combs = all_combs * factor
            np.random.shuffle(all_combs)

            total = len(all_combs)
            test_split = int(total * test_ratio)
            val_split = test_split + int(total * val_ratio)

            for i, comb in enumerate(all_combs):
                key = (mid, comb)
                if i < test_split:
                    split_dict[key] = "test"
                elif i < val_split:
                    split_dict[key] = "val"
                else:
                    split_dict[key] = "train"

    return split_dict


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", "--name", type=str, default="mvBCNN_new")
    parser.add_argument("-bs", "--batchSize", type=int, default=16)
    parser.add_argument("-num_two", type=int, default=8)  # 保留参数位
    parser.add_argument("-num_val", type=int, default=2000000)  # 保留参数位
    parser.add_argument("-rate1", type=float, default=0.8)
    parser.add_argument("-rate2", type=float, default=0.2)
    parser.add_argument("-nstop", type=int, default=8)
    parser.add_argument("-lr", type=float, default=5e-5)
    parser.add_argument("-weight_decay", type=float, default=1e-4)
    parser.add_argument("-no_pretraining", dest="no_pretraining", action="store_true")
    parser.add_argument("-BCNN_name", "--BCNN_name", type=str, default="resnet14")
    parser.add_argument("-num_views", type=int, default=2)
    parser.add_argument("-test_mode", action="store_true")  # 保留参数位
    parser.add_argument("-root_dir", type=str, default="")

    # 你指定的 CSV
    parser.add_argument(
        "--csv_path",
        type=str,
        default="",
    )

    args = parser.parse_args()
    pretraining = not args.no_pretraining

    loss_fn_h1 = nn.CrossEntropyLoss()
    loss_fn_h2 = nn.CrossEntropyLoss()

    experiment_id = str(uuid.uuid4())
    result_root = "/internfs/pengqianwen/MVBBCNN/results"
    log_dir = os.path.join(result_root, args.name, experiment_id)
    create_folder(log_dir)

    with open(os.path.join(log_dir, "config.json"), "w") as config_f:
        json.dump(vars(args), config_f, indent=4)

    # 读取 CSV 并构建映射
    spacegroup_to_materials, material_id_dict = load_materials_by_spacegroup(
        csv_path=args.csv_path,
        root_dir=args.root_dir,
    )

    split_dict = split_by_spacegroup(
        spacegroup_to_materials=spacegroup_to_materials,
        root_dir=args.root_dir,
        material_id_dict=material_id_dict,
        num_views=args.num_views,
        seed=12,
        val_ratio=0.15,
        test_ratio=0.15,
        max_combs_per_material=10,
        oversample_crystals=("cubic", "triclinic"),
        oversample_factor=1,
    )

    print(f"数据集划分完成，共 {len(split_dict)} 个样本组合，包含 {len(spacegroup_to_materials)} 个空间群")

    transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ---------------- STAGE 1 (SVBCNN) ----------------
    stage_1_log_dir = os.path.join(log_dir, "stage_1")
    create_folder(stage_1_log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    cnet = SVBCNN(args.name, pretraining=pretraining, BCNN_name=args.BCNN_name)
    cnet.to(device)
    if torch.cuda.device_count() > 1:
        cnet = nn.DataParallel(cnet)

    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        # 如果你要开多卡可在这里加 DataParallel

    optimizer_stage1 = optim.Adam(cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_dataset = SingleImgDataset(root_dir=args.root_dir, split="train", split_dict=split_dict, transform=transform_train)
    val_dataset = SingleImgDataset(root_dir=args.root_dir, split="val", split_dict=split_dict)
    test_dataset = SingleImgDataset(root_dir=args.root_dir, split="test", split_dict=split_dict)

    print(f"Stage1 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=4, drop_last=True, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4, drop_last=True, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4, drop_last=False, pin_memory=True
    )

    trainer_stage1 = ModelNetTrainer(
        cnet, train_loader, val_loader, optimizer_stage1,
        loss_fn_h1, loss_fn_h2, "svBCNN", stage_1_log_dir,
        num_views=1, rate1=args.rate1, rate2=args.rate2, nstop=args.nstop
    )
    trainer_stage1.train(100)

    # ---------------- STAGE 2 (MVBCNN) ----------------
    stage_2_log_dir = os.path.join(log_dir, "stage_2")
    create_folder(stage_2_log_dir)

    cnet_2 = MVBCNN(args.name, cnet, BCNN_name=args.BCNN_name, num_views=args.num_views)
    cnet_2.to(device)
    if torch.cuda.device_count() > 1:
        cnet_2 = nn.DataParallel(cnet_2)
    del cnet

    optimizer_stage2 = optim.Adam(
        filter(lambda p: p.requires_grad, cnet_2.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_dataset = MultiviewImgDataset(root_dir=args.root_dir, split="train", split_dict=split_dict, transform=transform_train)
    val_dataset = MultiviewImgDataset(root_dir=args.root_dir, split="val", split_dict=split_dict)
    test_dataset = MultiviewImgDataset(root_dir=args.root_dir, split="test", split_dict=split_dict)

    print(f"Stage2 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=4,
        drop_last=True, pin_memory=True, collate_fn=mv_collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4,
        drop_last=True, pin_memory=True, collate_fn=mv_collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4,
        drop_last=False, pin_memory=True, collate_fn=mv_collate_fn
    )

    with open(os.path.join(log_dir, "config_2.json"), "w") as config_f:
        json.dump({}, config_f, indent=4)

    trainer_stage2 = ModelNetTrainer(
        cnet_2, train_loader, val_loader, optimizer_stage2,
        loss_fn_h1, loss_fn_h2, "mvBCNN", stage_2_log_dir,
        num_views=args.num_views, rate1=args.rate1, rate2=args.rate2, nstop=args.nstop
    )
    trainer_stage2.train(100)
