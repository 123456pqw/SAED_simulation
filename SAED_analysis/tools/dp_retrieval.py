import os
import io
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import sys
sys.path.append('/share')
from SAED_analysis.models.mvbcnn import SVCNN, MVCNN
import re
from pymatgen.core import Composition
from typing import Union, List, Dict, Optional, Tuple
import tempfile
from starlette.datastructures import UploadFile

def load_pretrained_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    new_state = {}
    for k, v in state_dict.items():
        new_k = k[len("net."):] if k.startswith("net.") else k
        new_state[new_k] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"✅ Loaded weights with {len(missing)} missing and {len(unexpected)} unexpected keys.")

# ============= 初始化模型 ================= #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
svcnn = SVCNN("GVCNN", pretraining=False, cnn_name="resnet18")
mvcnn = MVCNN("GVCNN", svcnn)
load_pretrained_weights(mvcnn, "/share/SAED_analysis/models_pt/mvbcnn/model-00023.pth") 
mvcnn.to(device)
mvcnn.eval()

# ============= 数据预处理 ================= #
def extract_number(spacegroup_str):
    """从空间群字符串中提取数字"""
    match = re.search(r"'number': (\d+)", spacegroup_str)
    if match:
        return int(match.group(1))
    else:
        return None

# 加载数据并创建字典
MPdata = pd.read_csv('/share/SAED_analysis/utils/file_id.csv', sep=';', header=0, index_col=None)
material_id_dict = dict(zip(MPdata['material_id'], MPdata['crystal_system']))
crystal_system_dict = MPdata.groupby('crystal_system')['material_id'].apply(list).to_dict()
material_e_dict = dict(zip(MPdata['material_id'], MPdata['elements']))
material_formula_dict = dict(zip(MPdata['material_id'], MPdata['pretty_formula']))
formula_dict = MPdata.groupby('pretty_formula')['material_id'].apply(list).to_dict()

# 创建空间群相关字典
material_number_dict = {}
number_dict = {}
number_list = []

for material_id, spacegroup_str in zip(MPdata['material_id'], MPdata['spacegroup']):
    number = extract_number(spacegroup_str)
    if number is not None:
        number_list.append(number)
        material_number_dict[material_id] = number
    if number not in number_dict:
        number_dict[number] = []
    number_dict[number].append(material_id)

# 元素周期表顺序
all_elements = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
    'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
    'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
    'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Te'
]

element_order = {el: idx for idx, el in enumerate(all_elements)}

def clean_element_name(e):
    """清理元素名称"""
    return e.replace("Element", "").strip(" []',\"")

def parse_elements(elements):
    """解析元素列表"""
    if isinstance(elements, str):
        elements = re.findall(r'Element\s+([A-Za-z]+)', elements)
    elif isinstance(elements, list):
        elements = [clean_element_name(e) for e in elements]
    else:
        elements = []
    return elements

def sort_elements(element_list):
    """按元素周期表排序"""
    return sorted(element_list, key=lambda x: element_order.get(x, -1))

# 构建元素字典
crystal_e_dict = {}
for material_id, elements in material_e_dict.items():
    parsed = parse_elements(elements)
    sorted_elements = tuple(sort_elements(parsed))
    
    if sorted_elements not in crystal_e_dict:
        crystal_e_dict[sorted_elements] = []
    crystal_e_dict[sorted_elements].append(material_id)

def parse_elements_from_formula(user_input):
    """解析化学式或元素列表"""
    if isinstance(user_input, (list, tuple)):
        return sorted(set([x.strip() for x in user_input if x.strip()]))
    
    s = user_input.strip()
    if not s:
        return []
    
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return sorted(set(parts))
    
    try:
        comp = Composition(s)
        return sorted([el.symbol for el in comp.elements])
    except Exception:
        return [s]

# 创建新的元素字典
new_material_e_dict = {}
for mid, formula in material_formula_dict.items():
    try:
        new_material_e_dict[mid] = parse_elements_from_formula(formula)
    except:
        new_material_e_dict[mid] = []

# ============= 图像处理和特征提取 ================= #
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_image_data(img_input):
    """加载图像数据，支持多种输入类型"""
    if isinstance(img_input, np.ndarray):
        # numpy数组
        img = Image.fromarray(img_input).convert("RGB")
    elif isinstance(img_input, bytes):
        # 二进制数据
        img = Image.open(io.BytesIO(img_input)).convert("RGB")
    elif isinstance(img_input, UploadFile) or hasattr(img_input, 'file'):
        # UploadFile或Gradio文件对象
        if hasattr(img_input, 'name'):
            img = Image.open(img_input.name).convert("RGB")
        else:
            contents = img_input.file.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")
    elif isinstance(img_input, tempfile._TemporaryFileWrapper):
        # Gradio临时文件
        img = Image.open(img_input.name).convert("RGB")
    else:
        # 文件路径
        img = Image.open(img_input).convert("RGB")
    
    return img

@torch.no_grad()
def extract_feature_from_two_images(img1_input, img2_input, model):
    """提取两张图片的特征，支持多种输入类型"""
    imgs = []
    
    for img_input in [img1_input, img2_input]:
        img = load_image_data(img_input)
        img = img.crop((50, 50, 350, 350)).resize((1024, 1024), Image.BICUBIC)
        img = transform(img)
        imgs.append(img)
    
    batch = torch.stack(imgs).to(device)
    feat = model.extract_feature(batch, mode='task2', with_normalize=True)
    
    return feat.cpu().numpy().reshape(-1)

# ============= 特征数据库加载 ================= #
def load_feature_database_from_pt(
        gallery_pt,
        material_number_dict,
        material_e_dict,
        material_formula_dict,
        filter_elements=None,
        filter_sg=None,
        filter_formula=None
    ):
    """加载特征数据库并应用过滤"""
    gallery_bank = torch.load(gallery_pt, map_location="cpu")
    db_entries = []
    feat=[]
    for mid, feat_tensor in gallery_bank.items():
        sg = material_number_dict.get(mid, None)
        formula = material_formula_dict.get(mid, "")
        elements_raw = material_e_dict.get(mid, [])
        
        if sg is None:
            continue
        
        # 空间群过滤
        if filter_sg and int(sg) != int(filter_sg):
            continue
        
        # 化学式过滤
        if filter_formula and filter_formula.lower() != formula.lower():
            continue
        
        # 元素过滤
        if filter_elements:
            if isinstance(filter_elements, tuple) and len(filter_elements) == 2:
                user_input, mode = filter_elements
            else:
                user_input, mode = filter_elements, "contains"
            
            query_e = parse_elements_from_formula(user_input)
            query_e = sort_elements(query_e)
            se = set(elements_raw)
            
            if mode == "strict" and se != set(query_e):
                continue
            elif mode == "contains" and not set(query_e).issubset(se):
                continue
            elif mode == "subset" and not se.issubset(set(query_e)):
                continue
        
        # 处理特征
        feat = feat_tensor.numpy() if torch.is_tensor(feat_tensor) else np.array(feat_tensor)
        
        db_entries.append({
            "mid": mid,
            "sg": sg,
            "formula": formula,
            "elements": elements_raw,
            "feat": feat
        })

    '''
    db_entries.append({
            "mid": "mp-test",
            "sg": 14,
            "formula": "WO3",
            "elements": "W,O",
            "feat": np.random.rand(256)
        })
    '''
    print(f"✔ Loaded {len(db_entries)} feature entries from {gallery_pt}")
    return db_entries

# ============= 相似度搜索 ================= #
def search_top_k(query_feat, database, top_k=10):
    """搜索最相似的Top-K结果"""
    q = query_feat / np.linalg.norm(query_feat)
    results = []
    
    for item in database:
        f = item["feat"]
        f = f / np.linalg.norm(f)
        score = np.dot(q, f)
        results.append((score, item))
    
    results = sorted(results, key=lambda x: -x[0])[:top_k]
    
    return [{
        "mid": r[1]["mid"],
        "space_group": r[1]["sg"],
        "formula": r[1]["formula"],
        "elements": r[1]["elements"],
        #"similarity": float(r[0])
    } for r in results]

# ============= 主检索函数 ================= #
def retrieve_for_two_images(
        img1, img2, model=mvcnn,
        gallery_pt="/share/SAED_analysis/features_triplet/gallery_bank_mp.pt",
        material_number_dict=material_number_dict,
        material_e_dict=new_material_e_dict,
        material_formula_dict=material_formula_dict,
        filter_elements=None,
        filter_sg=None,
        filter_formula=None,
        top_k=10
    ):
    """
    主检索函数，支持多种输入类型
    
    参数:
        img1, img2: 图像路径、numpy数组、二进制数据、UploadFile等
        model: MVCNN模型
        gallery_pt: 特征库路径
        filter_elements: 元素过滤条件
        filter_sg: 空间群过滤
        filter_formula: 化学式过滤
        top_k: 返回结果数量
    """
    # 提取特征
    fused_feat = extract_feature_from_two_images(img1, img2, model)
    
    # 加载数据库
    database = load_feature_database_from_pt(
        gallery_pt=gallery_pt,
        material_number_dict=material_number_dict,
        material_e_dict=material_e_dict,
        material_formula_dict=material_formula_dict,
        filter_elements=filter_elements,
        filter_sg=filter_sg,
        filter_formula=filter_formula
    )
    
    # 检索结果
    return search_top_k(fused_feat, database, top_k)


# ============= 测试代码 ================= #
if __name__ == "__main__":
    # 测试文件路径输入
    results = retrieve_for_two_images(
        img1="/internfs/pengqianwen/MVBCNN/data/1/mp-9198/beam_0_0_1.png", 
        img2="/internfs/pengqianwen/MVBCNN/data/1/mp-9198/beam_0_1_0.png", 
        model=mvcnn,
        gallery_pt="/internfs/pengqianwen/MVBCNN/features_tripletv3/gallery_bank.pt",
        material_number_dict=material_number_dict,
        material_e_dict=new_material_e_dict,
        material_formula_dict=material_formula_dict,
        filter_elements=("W,O", "strict"),
        top_k=10
    )
    
    print(pd.DataFrame(results))