import os
import numpy as np
from collections import defaultdict
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tools.run_saed import run_ed_simulation
from torch.utils.data import DataLoader, Dataset
from models.mvbcnn import SVCNN, MVCNN
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import Union, Optional, List, Dict
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import cv2
from starlette.datastructures import UploadFile
import tempfile
import argparse
import os
import glob
import argparse
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.diffraction.tem import TEMCalculator
import numpy as np
import cv2
import sys
sys.path.append('/share/SAED_analysis')
from utils.picprocess import *

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SVCNN("GVCNN", pretraining=False, cnn_name="resnet18")
model.load_state_dict(torch.load("/share/SAED_analysis/models_pt/svbcnn/model-00083.pth"))
model.to(device)
model.eval() 

transform = transforms.Compose([
    #transforms.CenterCrop(300),
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def add_tem_noise_image(img,
                        poisson_scale=30,
                        gaussian_sigma=2):
    """
    img: uint8 or float image, shape (H, W) or (H, W, 3)
    return: noisy uint8 image
    """
    if img.ndim == 3:
        img = img[:, :, 0]  # 灰度

    img = img.astype(np.float32)
    img = img / (img.max() + 1e-8)

    # Poisson noise (electron counting)
    img_scaled = img * poisson_scale
    noisy = np.random.poisson(img_scaled)

    # Gaussian readout noise
    noisy = noisy + np.random.normal(
        0, gaussian_sigma, noisy.shape
    )

    noisy = np.clip(noisy, 0, None)
    noisy = noisy / (noisy.max() + 1e-8)
    noisy = (noisy * 255).astype(np.uint8)

    return noisy

def simulate_tem(cif_path, save_dir="./output",
                 beam_direction=(0, 0, 1), symprec=0.5):

    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"❌ CIF file not found: {cif_path}")

    os.makedirs(save_dir, exist_ok=True)

    from ase.io import read
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = read(cif_path)
    struct = AseAtomsAdaptor.get_structure(atoms)
    structure_std = struct

    tem_calc = TEMCalculator(
        beam_direction=beam_direction,
        camera_length=260,
        cs=0.8
    )

    # ===== simulate clean =====
    fig = tem_calc.get_plot_2d(structure_std)

    filename = os.path.splitext(os.path.basename(cif_path))[0]
    clean_path = os.path.join(save_dir, f"{filename}.png")
    noisy_path = os.path.join(save_dir, f"{filename}_noisy.png")

    fig.write_image(clean_path)

    # ===== reload =====
    img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("Failed to load generated TEM image.")

    # ===== noise =====
    noisy_img = add_tem_noise_image(
        img,
        poisson_scale=70,
        gaussian_sigma=2
    )

    cv2.imwrite(noisy_path, noisy_img)

    # ⭐⭐ 关键：返回路径
    return clean_path, noisy_path

def load_image(image_path_or_array, transform):
    """支持文件路径或numpy数组作为输入"""
    if isinstance(image_path_or_array, np.ndarray):
        # 直接处理numpy数组
        print(f"Loading image from numpy array: {image_path_or_array.shape}")
        im = Image.fromarray(image_path_or_array)
    else:
        print(f"Loading image: {image_path_or_array}")
        im = Image.open(image_path_or_array).convert('RGB')
    
    im = im.crop((50, 50, 350, 350)).resize((1024, 1024), Image.BICUBIC)
    im = transform(im)
    return im

def load_image_x(image_path_or_array, transform):
    """支持文件路径或numpy数组作为输入"""
    if isinstance(image_path_or_array, np.ndarray):
        # 直接处理numpy数组
        print(f"Loading image from numpy array: {image_path_or_array.shape}")
        im = Image.fromarray(image_path_or_array)
    else:
        print(f"Loading image: {image_path_or_array}")
        im = Image.open(image_path_or_array).convert('RGB')
    
    im = im.resize((1024, 1024), Image.BICUBIC)
    im = transform(im)
    return im

class ImageDataset(Dataset):
    def __init__(self, image_paths_or_arrays, transform):
        self.image_paths_or_arrays = image_paths_or_arrays
        self.transform = transform

    def __len__(self):
        return len(self.image_paths_or_arrays)

    def __getitem__(self, idx):
        item = self.image_paths_or_arrays[idx]
        
        # 如果是numpy数组，直接处理
        if isinstance(item, np.ndarray):
            im = load_image(item, self.transform)
            return im
        
        # 否则作为文件路径处理
        image_path = item
        if not os.path.isfile(image_path):
            raise ValueError(f"Path {image_path} is not a valid file.")
        
        im = load_image(image_path, self.transform)
        return im
    
class ImageDataset_x(Dataset):
    def __init__(self, image_paths_or_arrays, transform):
        self.image_paths_or_arrays = image_paths_or_arrays
        self.transform = transform
        self.to_pil = transforms.ToPILImage()

    def __len__(self):
        return len(self.image_paths_or_arrays)

    def __getitem__(self, idx):
        item = self.image_paths_or_arrays[idx]
        
        # 如果是numpy数组，直接处理
        if isinstance(item, np.ndarray):
            im = load_image_x(item, self.transform)
        else:
            # 否则作为文件路径处理
            image_path = item
            if not os.path.isfile(image_path):
                raise ValueError(f"Path {image_path} is not a valid file.")
            im = load_image_x(image_path, self.transform)

        # Convert tensor to PIL image for saving
        pil_image = self.to_pil(im).convert('RGB')
        pil_image = pil_image.convert('L')
        
        # 生成保存路径（如果是数组则使用临时名称）
        if isinstance(item, np.ndarray):
            save_path = f'numpy_array_processed.png'
        else:
            base_name = os.path.basename(item)
            file_name, ext = os.path.splitext(base_name)
            save_path = os.path.join(f'{file_name}_processed{ext}')
        
        print(f"Saved processed image to: {save_path}")
        return im

def get_image_features_batch(image_paths_or_arrays, model, transform, batch_size=32):
    dataset = ImageDataset(image_paths_or_arrays, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    features = []
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)  # Move data to GPU if available
            feature = model.extract_feature(inputs)  # Forward pass to get features
            feature = feature.squeeze()
            features.append(feature.cpu().numpy())  # Convert to numpy and collect features
    
    return np.vstack(features)  # Stack the features into a single 2D array

def get_image_features_x(image_paths_or_arrays, model, transform, batch_size=32):
    dataset = ImageDataset_x(image_paths_or_arrays, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    features = []
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)  # Move data to GPU if available
            feature = model.extract_feature(inputs, mode='task2')  # Forward pass to get features
            feature = feature.squeeze()
            features.append(feature.cpu().numpy())  # Convert to numpy and collect features
    
    return np.vstack(features)  # Stack the features into a single 2D array

def generate_beam_directions():
    directions = [[0, 0, 1]]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i == j == k == 0:
                    continue
                vec = np.array([i, j, k])
                angles = [
                    np.arccos(np.clip(np.dot(vec/np.linalg.norm(vec), np.array(d)/np.linalg.norm(d)), -1.0, 1.0))
                    for d in directions
                ]
                if np.min(angles) > 1e-3:
                    directions.append([i, j, k])
    return directions

# ============================
# 2. 对称性分组
# ============================
def group_beams_by_symmetry(cif_path, beams):
    structure = Structure.from_file(cif_path)
    sga = SpacegroupAnalyzer(structure)
    recip = structure.lattice.reciprocal_lattice
    
    beam_groups = defaultdict(list)
    
    for beam in beams:
        frac_beam = recip.get_fractional_coords(beam)
        standard_beam = None
        
        for op in sga.get_symmetry_operations():
            transformed = op.operate(frac_beam)
            cart_beam = recip.get_cartesian_coords(transformed).round(3)
            int_beam = np.round(cart_beam).astype(int)
            
            if standard_beam is None or tuple(int_beam) < tuple(standard_beam):
                standard_beam = tuple(int_beam)

        beam_groups[standard_beam].append(beam)
    
    return beam_groups

# ============================
# 辅助函数
# ============================
def _process_cif_input(cif_input) -> str:
    """处理CIF输入，返回文件路径"""
    if isinstance(cif_input, UploadFile):
        # 处理UploadFile类型
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.cif')
        contents = cif_input.file.read()
        temp_file.write(contents)
        temp_file.close()
        return temp_file.name
        
    elif isinstance(cif_input, bytes):
        # 处理二进制数据
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.cif')
        temp_file.write(cif_input)
        temp_file.close()
        return temp_file.name
        
    elif isinstance(cif_input, str):
        # 已经是文件路径
        return cif_input
        
    else:
        raise TypeError(f"CIF输入不支持的类型：{type(cif_input)}")

def _prepare_image_input(img_input):
    """准备图像输入，可以是路径、numpy数组、UploadFile等"""
    if isinstance(img_input, UploadFile):
        # 处理UploadFile类型的图像
        contents = img_input.file.read()
        image_pil = Image.open(BytesIO(contents)).convert("RGB")
        return np.array(image_pil)
        
    elif isinstance(img_input, bytes):
        # 处理二进制数据
        image_pil = Image.open(BytesIO(img_input)).convert("RGB")
        return np.array(image_pil)
        
    elif isinstance(img_input, str):
        # 文件路径，直接返回
        return img_input
        
    elif isinstance(img_input, np.ndarray):
        # numpy数组，直接返回
        return img_input
        
    else:
        raise TypeError(f"图像输入不支持的类型：{type(img_input)}")

# ============================
# 3. 主函数：模拟 + 特征匹配
# ============================
'''
async def match_experiment_with_simulation(
    exp_img_input: Union[np.ndarray, str, bytes, UploadFile],
    cif_input: Union[str, UploadFile, bytes],
    save_root: str = "./saed_sim_match/",
    top_k: int = 5,
    regenerate: bool = False,
    delete_temp: bool = True
) -> List[Dict]:
    """
    输入：
        exp_img_input: 实验 SAED 图（支持路径、numpy数组、UploadFile、二进制数据）
        cif_input: CIF文件（支持路径、UploadFile、二进制数据）
        save_root: 保存根目录
        top_k: 返回前k个匹配结果
        regenerate: 是否重新运行模拟
        delete_temp: 是否删除临时文件

    输出：
        返回最相似模拟图的列表
    """
    temp_files = []
    
    try:
        print("start matching...")
        
        # -------------------------
        # 处理输入数据
        # -------------------------
        # 处理CIF文件
        cif_path = _process_cif_input(cif_input)
        print(f"CIF file prepared at: {cif_path}")
        if isinstance(cif_input, (UploadFile, bytes)):
            temp_files.append(cif_path)
        
        # 准备图像输入
        prepared_img = _prepare_image_input(exp_img_input)
        print(1)
        # -------------------------
        # Step 1. 提取实验 SAED 特征
        # -------------------------
        exp_feat = get_image_features_x([prepared_img], model, transform)
        print(2)
        # -------------------------
        # Step 2. 生成晶向 + 对称性分组
        # -------------------------
        raw_beams = generate_beam_directions()
        beam_groups = group_beams_by_symmetry(cif_path, raw_beams)

        structure = Structure.from_file(cif_path)
        sga = SpacegroupAnalyzer(structure)
        sg_info = f"{sga.get_space_group_number()}_{sga.get_space_group_symbol()}"

        sim_dir = os.path.join(save_root, sg_info)
        os.makedirs(sim_dir, exist_ok=True)
        print(3)
        sim_paths = []

        # -------------------------
        # Step 3. 遍历每个对称等价组 → 只模拟一次
        # -------------------------
        for group_key, beams in beam_groups.items():
            group_str = "_".join(map(str, group_key))
            group_path = os.path.join(sim_dir, f"group_{group_str}")
            os.makedirs(group_path, exist_ok=True)

            sim_png = os.path.join(group_path, "sim.png")

            # 需要模拟

            if regenerate or not os.path.exists(sim_png):
                clean_path, noisy_path = simulate_tem(
                    cif_path,
                    save_dir=os.path.dirname(sim_png),
                    beam_direction=tuple(group_key)   # ← zone axis → beam
                )

                sim_png = noisy_path   # ⭐ 如果你想用 noisy 作为检索图
                print(f"Simulated and saved: {sim_png}")

            sim_paths.append(sim_png)

        # -------------------------
        # Step 4. 提取所有模拟图特征
        # -------------------------
        sim_feats = get_image_features_batch(sim_paths, model, transform)
        print(4)
        # -------------------------
        # Step 5. 计算相似度（余弦）
        # -------------------------
        cos_scores = np.dot(exp_feat, sim_feats.T).flatten()

        top_idx = np.argsort(cos_scores)[::-1][:top_k]

        # -------------------------
        # Step 6. 汇总结果
        # -------------------------
        results = []
        for i in top_idx:
            results.append({
                "simulation_image": sim_paths[i],
                "similarity": float(cos_scores[i]),
                "spacegroup_info": sg_info,
                "rank": int(np.where(top_idx == i)[0][0] + 1)
            })
        print(results)
            
        return results
        
    finally:
        # 清理临时文件
        if delete_temp:
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
'''

# =====================================================
# 主函数
# =====================================================
async def match_experiment_with_simulation(

    exp_img_input,
    cif_input,
    exp_img_type: str = "png",
    save_root: str = "./saed_sim_match/",
    top_k: int = 5,
    regenerate: bool = False,
    delete_temp: bool = True

) -> List[Dict]:

    temp_files = []

    try:

        print("================================")
        print("start matching...")
        print("================================")

        # =====================================================
        # CIF处理
        # =====================================================
        cif_path, temp_cif = await process_cif_input(cif_input)

        if temp_cif:
            temp_files.append(temp_cif)

        # =====================================================
        # 图像处理（关键：统一入口）
        # =====================================================
        prepared_img = await decode_image_to_np(
            exp_img_input
        )

        print("image prepared")

        # =====================================================
        # feature
        # =====================================================
        exp_feat = np.asarray(
            get_image_features_x([prepared_img], model, transform)
        )

        # =====================================================
        # symmetry
        # =====================================================
        raw_beams = generate_beam_directions()
        beam_groups = group_beams_by_symmetry(cif_path, raw_beams)

        structure = Structure.from_file(cif_path)
        sga = SpacegroupAnalyzer(structure)

        sg_symbol = str(sga.get_space_group_symbol()).encode(
            "utf-8", errors="ignore"
        ).decode("utf-8")

        sg_info = f"{int(sga.get_space_group_number())}_{sg_symbol}"

        sim_dir = os.path.join(save_root, sg_info)
        os.makedirs(sim_dir, exist_ok=True)

        sim_paths = []

        # =====================================================
        # simulation
        # =====================================================
        for group_key, beams in beam_groups.items():

            group_str = "_".join(map(str, group_key))
            group_path = os.path.join(sim_dir, f"group_{group_str}")
            os.makedirs(group_path, exist_ok=True)

            sim_png = os.path.join(group_path, "sim.png")

            if regenerate or not os.path.exists(sim_png):

                clean_path, noisy_path = simulate_tem(
                    cif_path,
                    save_dir=os.path.dirname(sim_png),
                    beam_direction=tuple(group_key)
                )

                sim_png = noisy_path

            # 🔥 强制 string + safe
            sim_paths.append(str(sim_png))

        # =====================================================
        # feature sim
        # =====================================================
        sim_feats = np.asarray(
            get_image_features_batch(sim_paths, model, transform)
        )

        # =====================================================
        # similarity
        # =====================================================
        cos_scores = np.dot(exp_feat, sim_feats.T).flatten()

        top_idx = np.argsort(cos_scores)[::-1][:top_k]

        # =====================================================
        # results
        # =====================================================
        results = []

        for rank_id, i in enumerate(top_idx):

            results.append({
                "simulation_image": str(sim_paths[int(i)]),
                "similarity": float(cos_scores[int(i)]),
                "spacegroup_info": str(sg_info),
                "rank": int(rank_id + 1)
            })

        # =====================================================
        # 🔥 FINAL SAFE RETURN（关键）
        # =====================================================
        safe_results = safe_output(results)

        json.dumps(safe_results, ensure_ascii=False)

        return safe_results

    finally:

        if delete_temp:
            for f in temp_files:
                try:
                    os.unlink(f)
                except:
                    pass


import asyncio
async def main():
    results = await match_experiment_with_simulation(
        #exp_img_input='/internfs/pengqianwen/MVBCNN_SAED/WechatIMG7659.jpg',
        exp_img_input='/internfs/pengqianwen/MVBCNN/SAED/AgVP2S6/AgVP2S6_0011.png',
        cif_input='/internfs/pengqianwen/MVBCNN/SAED/AgVP2S6/VAg(PS3)2.cif'
    )
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
