import torch
import os
import random
from PIL import Image
from torchvision import transforms
import re
import time
import json
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import argparse
import shutil
import torch
import os
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from models.MVBCNN import SVBCNN, MVBCNN
import os
import random
import torch
from PIL import Image

def extract_number(spacegroup_str):
    # 匹配 'number': 后面的数字
    match = re.search(r"'number': (\d+)", spacegroup_str)
    if match:
        return int(match.group(1))
    else:
        return None  # 如果没有找到 'number'，返回 None

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="GVBCNN")
parser.add_argument("-BCNN_name", "--BCNN_name", type=str, help="BCNN model name", default="inception")
args = parser.parse_args()

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MPdata = pd.read_csv('', sep=';', header=0, index_col=None)

# Create dictionaries
material_id_dict = dict(zip(MPdata['material_id'], MPdata['crystal_system']))
crystal_system_dict = MPdata.groupby('crystal_system')['material_id'].apply(list).to_dict()
material_e_dict = dict(zip(MPdata['material_id'], MPdata['elements']))
material_formula_dict = dict(zip(MPdata['material_id'], MPdata['pretty_formula']))
formula_dict= MPdata.groupby('pretty_formula')['material_id'].apply(list).to_dict()

# 使用提取的 'number' 和 material_id 创建字典
material_number_dict = {}
number_dict = {}
number_list = []
# 遍历 DataFrame 中的每一行
for material_id, spacegroup_str in zip(MPdata['material_id'], MPdata['spacegroup']):
    number = extract_number(spacegroup_str)
    print(number)
    if number is not None:
        number_list.append(number)
        material_number_dict[material_id] = number
    if number not in number_dict:
        number_dict[number] = []
    number_dict[number].append(material_id)

# All possible elements in order
all_elements = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
    'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
    'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
    'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Te'
]

# Create a dictionary to order elements
element_order = {el: idx for idx, el in enumerate(all_elements)}

import re

def clean_element_name(e):
    """去掉 'Element ' 前缀并去除空格"""
    return e.replace("Element", "").strip(" []',\"")

def parse_elements(elements):
    """兼容字符串或列表输入"""
    if isinstance(elements, str):
        # 提取所有形如 'Element XX' 的部分
        elements = re.findall(r'Element\s+([A-Za-z]+)', elements)
    elif isinstance(elements, list):
        # 直接清洗
        elements = [clean_element_name(e) for e in elements]
    else:
        elements = []
    return elements

def sort_elements(element_list):
    """按元素周期表顺序排序"""
    return sorted(element_list, key=lambda x: element_order.get(x, -1))

# --- 构建 crystal_e_dict ---
crystal_e_dict = {}
for material_id, elements in material_e_dict.items():
    parsed = parse_elements(elements)
    sorted_elements = tuple(sort_elements(parsed))
    #print(f"原始: {elements} -> 提取后: {parsed} -> 排序后: {sorted_elements}")

    if sorted_elements not in crystal_e_dict:
        crystal_e_dict[sorted_elements] = []
    crystal_e_dict[sorted_elements].append(material_id)

# 复用相同 transform
transform = transforms.Compose([
    transforms.CenterCrop(300),
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pretrained_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    new_state = {}
    for k, v in state_dict.items():
        new_k = k[len("net."):] if k.startswith("net.") else k
        new_state[new_k] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"✅ Loaded weights with {len(missing)} missing and {len(unexpected)} unexpected keys.")


# 加载微调好的模型
svBCNN = SVBCNN("GVBCNN", pretraining=False, BCNN_name="resnet18")
mvBCNN = MVBCNN("GVBCNN", svBCNN)
load_pretrained_weights(mvBCNN, "") 
mvBCNN.to(device)
mvBCNN.eval()

def extract_feature_for_material(mat_dir, num_views=2, seed=42):
    allowed_beams = {
        "beam_0_0_1.png",
        "beam_1_0_0.png",
        "beam_0_1_0.png"
    }

    imgs = [
        os.path.join(mat_dir, f)
        for f in os.listdir(mat_dir)
        if f in allowed_beams
    ]

    if len(imgs) == 0:
        return None

    # --------------------------------------------------
    # 根据 material path + seed 创建独立随机数生成器
    # 保证：
    # 1. 同一个 material + 同一个 seed -> 选择结果一致
    # 2. 不同 material -> 可以选择不同 beam
    # 3. 不影响程序其他地方的 random 状态
    # --------------------------------------------------
    material_name = os.path.basename(os.path.normpath(mat_dir))
    local_seed = hash((material_name, seed)) & 0xffffffff
    rng = random.Random(local_seed)

    # 如果 beam 数量不足，允许重复
    if len(imgs) < num_views:
        imgs = imgs * (num_views // len(imgs) + 1)

    chosen = rng.sample(imgs, num_views)

    # 为了保证不同系统 os.listdir 顺序不会影响结果
    chosen = sorted(chosen)

    tensors = [
        transform(Image.open(p).convert("RGB"))
        for p in chosen
    ]

    x = torch.stack(tensors, dim=0).unsqueeze(0).to(device)  # [1,V,C,H,W]

    with torch.no_grad():
        feat = mvBCNN.extract_feature(
            x.view(-1, *x.shape[2:])
        )

    return feat.mean(dim=0).cpu().numpy()

# ========== 主循环 ==========
input_root = ""
output_root = ""
os.makedirs(output_root, exist_ok=True)


feature_bank = {}
for sg in sorted(os.listdir(input_root)):
    sg_dir = os.path.join(input_root, sg)
    if not os.path.isdir(sg_dir):
        continue

    print(f"Processing space group {sg} ...")

    for mid in tqdm(sorted(os.listdir(sg_dir))):
        mat_dir = os.path.join(sg_dir, mid)
        if not os.path.isdir(mat_dir):
            continue

        feat = extract_feature_for_material(mat_dir)
        if feat is None:
            continue

        feature_bank[mid] = feat

torch.save(feature_bank, os.path.join(output_root, "feature_bank.pt"))
print("✅ 所有特征已保存。")


gallery_bank = {}
query_bank = {}

for sg in sorted(os.listdir(input_root)):
    sg_dir = os.path.join(input_root, sg)
    if not os.path.isdir(sg_dir):
        continue

    for mid in sorted(os.listdir(sg_dir)):
        mat_dir = os.path.join(sg_dir, mid)
        if not os.path.isdir(mat_dir):
            continue

        # 两个方向分别提取
        feat1 = extract_feature_for_material(mat_dir, num_views=2)
        feat2 = extract_feature_for_material(mat_dir, num_views=2)

        if feat1 is None or feat2 is None:
            continue

        gallery_bank[mid] = feat1
        query_bank[mid] = feat2

torch.save(gallery_bank, os.path.join(output_root, "gallery_bank.pt"))
torch.save(query_bank, os.path.join(output_root, "query_bank.pt"))

# 显式允许 numpy 数组反序列化
torch.serialization.add_safe_globals([
    np.ndarray,
    np.dtype,
    np._core.multiarray._reconstruct,
    np.dtypes.Float32DType,
    np.dtypes.Float64DType,
    np.dtypes.Int64DType,
    np.dtypes.Int32DType,
])

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

gallery_bank = torch.load(os.path.join(output_root, "gallery_bank.pt"))
query_bank = torch.load(os.path.join(output_root, "query_bank.pt"))

def compute_recall_dual_banks(gallery_bank, query_bank, crystal_e_dict, k=1):
    recalls = []

    for elem_key, mids in tqdm(crystal_e_dict.items(), desc=f"Recall@{k}"):
        valid_mids = [mid for mid in mids if mid in gallery_bank and mid in query_bank]
        if len(valid_mids) < 2:
            continue

        # 构建矩阵
        G = np.stack([gallery_bank[mid] for mid in valid_mids])
        Q = np.stack([query_bank[mid] for mid in valid_mids])

        # 全矩阵相似度 [n x n]
        sims = cosine_similarity(Q, G)

        for i, mid in enumerate(valid_mids):
            sorted_idx = np.argsort(-sims[i])
            gt_idx = i
            rank = np.where(sorted_idx == gt_idx)[0][0] + 1
            recalls.append(rank <= k)

    recall = np.mean(recalls)
    print(f"✅ Recall@{k}: {recall:.3f} ({len(recalls)} samples)")
    return recall

compute_recall_dual_banks(gallery_bank, query_bank, crystal_e_dict, k=1)
#compute_recall_dual_banks(gallery_bank, query_bank, crystal_e_dict, k=5)
#compute_recall_dual_banks(gallery_bank, query_bank, crystal_e_dict, k=10)

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def compute_recall_same_elements(gallery_bank, query_bank, crystal_e_dict, k=1):
    recalls = []
    total = 0

    for elem_key, mids in tqdm(crystal_e_dict.items(), desc=f"Recall@{k} (same elements)"):
        # 只比较同元素组合内部
        valid_mids = [mid for mid in mids if mid in gallery_bank and mid in query_bank]
        if len(valid_mids) < 2:
            continue  # 至少需要两个样本才可检索

        # 构建特征矩阵
        G = np.stack([gallery_bank[mid] for mid in valid_mids])
        Q = np.stack([query_bank[mid] for mid in valid_mids])

        # L2 归一化（可选，但推荐）
        G = G / np.linalg.norm(G, axis=1, keepdims=True)
        Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)

        # 批量计算相似度
        sims = cosine_similarity(Q, G)

        # 计算每个查询的排名
        for i, mid in enumerate(valid_mids):
            sorted_idx = np.argsort(-sims[i])  # 降序
            gt_idx = i
            rank = np.where(sorted_idx == gt_idx)[0][0] + 1  # 从1开始
            recalls.append(rank <= k)
            total += 1

    recall = np.mean(recalls) if total > 0 else 0
    print(f"✅ Recall@{k} (same elements): {recall:.3f}  ({total} samples)")
    return recall



gallery_bank = torch.load("")
query_bank   = torch.load("")

compute_recall_same_elements(gallery_bank, query_bank, crystal_e_dict, k=1)
compute_recall_same_elements(gallery_bank, query_bank, crystal_e_dict, k=5)
compute_recall_same_elements(gallery_bank, query_bank, crystal_e_dict, k=10)
