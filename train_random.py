import numpy as np
import torch
import torch
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True  # 启用加速算法
import os
import torch.optim as optim
import torch.nn as nn
import random
import shutil
import json
import argparse
import uuid
import pandas as pd
from torchvision import transforms, datasets
import numpy as np
from collections import defaultdict
from tools.Trainer_2 import ModelNetTrainer
from tools.ImgDataset_random_aug import MultiviewImgDataset, SingleImgDataset
from models.MVBCNN import MVBCNN, SVBCNN
#from utils.simulate import generate_beam_directions, load_materials_by_spacegroup, extract_spacegroup_number
MPdata = pd.read_csv('', sep=';', header=0, index_col=None)
material_id_dict = dict(zip(MPdata['material_id'], MPdata['crystal_system']))
import torch
import re
# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="mvBCNN_new")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=16)
parser.add_argument("-num_two", type=int, help="number of models per class", default=8)
parser.add_argument("-num_val", type=int, help="number of val ", default=2000000)
parser.add_argument("-rate1", type=float, help="number of val ", default=0.8)
parser.add_argument("-rate2", type=float, help="number of val ", default=0.2)
parser.add_argument("-nstop", type=int, help="number of val ", default=8)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0001)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-BCNN_name", "--BCNN_name", type=str, help="BCNN model name", default="resnet14")
parser.add_argument("-num_views", type=int, help="number of views", default=2)
parser.add_argument("-test_mode",  action='store_true',help="run or not")#不加上就是使用数据增强
parser.add_argument("-root_dir", type=str, default="/internfs/pengqianwen/MVBBCNN/data2")
parser.set_defaults(train=False)

def extract_spacegroup_number(spacegroup_str):
    try:
        if pd.isna(spacegroup_str) or str(spacegroup_str).strip() in ('', 'None', '{}'):
            return None
        json_str = str(spacegroup_str).replace("'", '"')
        match = re.search(r'"number"\s*:\s*(\d+)', json_str)
        if match:
            number = match.group(1)
        match = re.search(r'number[&#39;"]?\s*:\s*(\d+)', json_str)
        if match:
            number = match.group(1)
        return number
    except:
        return None
    
def load_materials_by_spacegroup(csv_path, source_dir):
    df = pd.read_csv(csv_path, sep=';')
    spacegroup_to_materials = defaultdict(list)
    source_materials = {
        d[:-4] for d in os.listdir(source_dir)
        if d.startswith('mp-') or d.startswith('mvc-')
    }
    #print(source_materials)
    #print(len(source_materials))
    count=0
    for _, row in df.iterrows():
        mid = row['material_id']
        if mid not in source_materials:
            continue
        count += 1
        sg = extract_spacegroup_number(row['spacegroup'])
        if sg is not None:
            spacegroup_to_materials[sg].append(mid)
    #print(f"✅ 共找到 {count} 个材料的空间群信息")
    OUTPUT_CSV = "spacegroup_count_summary.csv"
    #save_summary(spacegroup_to_materials, OUTPUT_CSV)
    return spacegroup_to_materials

def create_folder(log_dir):
    """Create a folder for logging. If it exists, delete and recreate it."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    else:
        print('WARNING: Summary folder already exists! It will be overwritten!')
        shutil.rmtree(log_dir)
        os.makedirs(log_dir)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # tensor: (C, H, W)
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

'''
def split_by_spacegroup(spacegroup_to_materials, seed=42, 
                        val_ratio=0.15, test_ratio=0.15,
                        max_samples_per_group=5):
    """
    参数说明：
    - spacegroup_to_materials: 字典，格式 {空间群: [material_id列表]}
    - max_samples_per_group: 每个空间群最多保留的样本数（默认100）
    """
    split_dict = {}
    np.random.seed(seed)  # 全局随机种子

    for sg, materials in spacegroup_to_materials.items():
        # 步骤1：随机采样控制数量
        materials = sorted(materials)
        np.random.shuffle(materials)
        selected = materials[:max_samples_per_group]  # 截断至最大数量
        print(f"空间群 {sg} 选取样本数: {len(selected)}")
        # 步骤2：计算划分点
        total = len(selected)
        test_split = int(total * test_ratio)
        val_split = test_split + int(total * val_ratio)

        # 步骤3：分配标签
        for i, mid in enumerate(selected):
            if i < test_split:
                split_dict[mid] = 'test'
            elif i <  val_split:
                split_dict[mid] = 'val'
            else:
                split_dict[mid] = 'train'

    return split_dict
'''
def mv_collate_fn(batch):
    class_id_h1_list = []
    class_id_h2_list = []
    imgs_list = []
    paths_list = []

    for class_id_h1, class_id_h2, imgs, paths in batch:
        class_id_h1_list.append(class_id_h1)
        class_id_h2_list.append(class_id_h2)
        imgs_list.append(imgs)  # imgs: [V, C, H, W]
        
        paths_list.append(paths)
    #print("stacked imgs shape:", torch.stack(imgs_list).shape)

    # imgs_list: [B, V, C, H, W]
    return (
        torch.tensor(class_id_h1_list),
        torch.tensor(class_id_h2_list),
        torch.stack(imgs_list),  # final shape: [B, V, C, H, W]
        paths_list
    )

def split_by_spacegroup(spacegroup_to_materials, root_dir, num_views=2, seed=42, 
                        val_ratio=0.15, test_ratio=0.15, max_combs_per_material=10, oversample_crystals=('cubic', 'triclinic'),oversample_factor=1):
    """
    参数说明：
    - spacegroup_to_materials: 字典，格式 {空间群: [material_id列表]}
    - max_samples_per_group: 每个空间群最多保留的样本数（默认100）
    """
    split_dict = {}
    np.random.seed(seed)  # 全局随机种子

    for sg, materials in spacegroup_to_materials.items():
        for mid in materials:
            material_dir = os.path.join(root_dir, str(sg), mid)
            if not os.path.isdir(material_dir):
                continue
            views = sorted([f for f in os.listdir(material_dir) if f.endswith('.png')])
            if len(views) < num_views:
                continue

            # 枚举所有组合
            from itertools import combinations
            all_combs = list(combinations(views, num_views))

            crystal_system = material_id_dict.get(mid, None)
            factor = oversample_factor if crystal_system in oversample_crystals else 1

            # 限制每个材料最多取多少个组合
            if max_combs_per_material is not None and len(all_combs) > max_combs_per_material:
                all_combs = random.sample(all_combs, max_combs_per_material)

            # 重复组合以实现多采样
            all_combs = all_combs * factor
            np.random.shuffle(all_combs)

            total = len(all_combs)
            test_split = int(total * test_ratio)
            val_split = test_split + int(total * val_ratio)

            for i, comb in enumerate(all_combs):
                key = (mid, comb)
                if i < test_split:
                    split_dict[key] = 'test'
                elif i < val_split:
                    split_dict[key] = 'val'
                else:
                    split_dict[key] = 'train'

    return split_dict

if __name__ == '__main__':
    args = parser.parse_args()
    pretraining = not args.no_pretraining
    
    loss_fn_h1 = nn.CrossEntropyLoss()
    loss_fn_h2 = nn.CrossEntropyLoss()
    
    # Generate a unique ID for the experiment
    experiment_id = str(uuid.uuid4())
    root_dir=''
    log_dir = os.path.join(root_dir, args.name, experiment_id)
    create_folder(log_dir)

    # Save the configuration
    config_f = open(os.path.join(log_dir, 'config.json'), 'w')
    json.dump(vars(args), config_f, indent=4)
    config_f.close()
    

    # STAGE 1
    stage_1_log_dir = os.path.join(log_dir, 'stage_1')
    create_folder(stage_1_log_dir)
    #device_ids = list(range(torch.cuda.device_count()))  
    cnet = SVBCNN(args.name, pretraining=pretraining, BCNN_name=args.BCNN_name)
   
    # 加载权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    cnet.to(device)
    if torch.cuda.device_count() > 1:  # 检查电脑是否有多块GPU
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        #cnet  = nn.DataParallel(cnet, device_ids=[0, 1, 2, 3,4,5,6,7])  # 将模型对象转变为多GPU并行运算的模型

    
    #cnet = torch.nn.DataParallel(cnet, device_ids=device_ids).to(device_ids[0])  
    
    
    optimizer = optim.Adam(cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    from sklearn.model_selection import train_test_split
    MPdata = pd.read_csv('', sep=';', header=0, index_col=None)
    spacegroup_to_materials=load_materials_by_spacegroup('')
    #split_dict = split_by_spacegroup(spacegroup_to_materials,max_samples_per_group=args.num_val, seed=12, val_ratio=0.15, test_ratio=0.15)
    split_dict = split_by_spacegroup(spacegroup_to_materials,args.root_dir, num_views=2,seed=12, val_ratio=0.15, test_ratio=0.15)
    
    # 数据增强 pipeline（用于训练）
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(1024, scale=(0.6, 1.2)),   # 随机缩放 + 裁剪
        transforms.RandomRotation(degrees=30),                 # 随机旋转 ±30°
        transforms.RandomHorizontalFlip(p=0.5),                # 随机水平翻转
        transforms.ToTensor(),
        AddGaussianNoise(0., 0.015),                            # 加入高斯噪声
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    # 测试/验证集只做标准化
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    print(f"数据集划分完成，共 {len(split_dict)} 个材料，包含 {len(spacegroup_to_materials)} 个空间群")
    train_dataset = SingleImgDataset(root_dir=args.root_dir, split='train', split_dict=split_dict,transform=transform_train)
    val_dataset = SingleImgDataset(root_dir=args.root_dir, split='val', split_dict=split_dict)
    test_dataset = SingleImgDataset(root_dir=args.root_dir, split='test', split_dict=split_dict)
    print(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}, 测试集样本数: {len(test_dataset)}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=4,drop_last=True,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4,drop_last=True,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

    config={}
    

    trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer,loss_fn_h1, loss_fn_h2, 'svBCNN', stage_1_log_dir, num_views=1,rate1=args.rate1,rate2=args.rate2,nstop=args.nstop)
    trainer.train(100)
    
    
    # STAGE 2
    #config={}
    stage_2_log_dir = os.path.join(log_dir, 'stage_2')
    create_folder(stage_2_log_dir)
    cnet_2 = MVBCNN(args.name, cnet, BCNN_name=args.BCNN_name, num_views=args.num_views)
    cnet_2.to(device)
    del cnet

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cnet_2.parameters()),lr=args.lr, weight_decay=args.weight_decay)
    
    train_dataset = MultiviewImgDataset(root_dir=args.root_dir, split='train', split_dict=split_dict,transform=transform_train)
    val_dataset = MultiviewImgDataset(root_dir=args.root_dir, split='val', split_dict=split_dict)
    test_dataset = MultiviewImgDataset(root_dir=args.root_dir, split='test', split_dict=split_dict)
    print(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}, 测试集样本数: {len(test_dataset)}")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=4,drop_last=True,pin_memory=True,collate_fn=mv_collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4,drop_last=True,pin_memory=True,collate_fn=mv_collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4, drop_last=False, pin_memory=True,collate_fn=mv_collate_fn)

 

    with open(os.path.join(log_dir, 'config_2.json'), 'w') as config_f:
        json.dump(config, config_f, indent=4)

    trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, loss_fn_h1, loss_fn_h2, 'mvBCNN', stage_2_log_dir, num_views=args.num_views,rate1=args.rate1,rate2=args.rate2,nstop=args.nstop)
    trainer.train(100)
