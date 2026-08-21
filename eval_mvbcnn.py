import numpy as np
import torch
import torch.nn as nn
import os
import json
import argparse
import pandas as pd
from torchvision import transforms
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tools.ImgDataset_random_aug import MultiviewImgDataset, SingleImgDataset
from models.MVBCNN import MVBCNN, SVBCNN
from utils.simulate import load_materials_by_spacegroup

# ====================== 配置参数 ======================
parser = argparse.ArgumentParser()
parser.add_argument("-root_dir", type=str, default="", help="数据根目录")
parser.add_argument("-batchSize", type=int, default=32, help="评估批次大小")
parser.add_argument("-num_views", type=int, default=3, help="多视图数量（需与训练时一致）")
parser.add_argument("-BCNN_name", type=str, default="resnet14", help="BCNN骨干网络名称（需与训练时一致）")
parser.add_argument("-save_dir", type=str, default="crystal_system_evaluation", help="评估结果保存目录")
parser.add_argument("-device", type=str, default="cuda", help="评估设备（cuda/cpu）")
args = parser.parse_args()

# 创建结果保存目录（按晶系拆分的结果单独存子文件夹）
os.makedirs(args.save_dir, exist_ok=True)
crystal_system_save_dir = os.path.join(args.save_dir, "crystal_system_details")
os.makedirs(crystal_system_save_dir, exist_ok=True)

# ====================== 设备配置 ======================
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
torch.backends.cudnn.benchmark = True  # 加速推理

# ====================== 核心映射关系加载（关键：材料ID→晶系） ======================
# 1. 加载材料ID-晶系-类别ID的完整映射（从原始CSV获取）
MPdata = pd.read_csv('/internfs/pengqianwen/stem-crystal-system/single_opinion/Data/file_id.csv', sep=';', header=0, index_col=None)
# 构建三大核心映射：
material_to_crystal = dict(zip(MPdata['material_id'], MPdata['crystal_system']))  # 材料ID→晶系名称
material_to_h1 = dict(zip(MPdata['material_id'], MPdata['class_id_h1']))          # 材料ID→H1类别ID（晶系对应的数字编码）
crystal_to_h1 = dict(zip(MPdata['crystal_system'], MPdata['class_id_h1']))        # 晶系名称→H1类别ID（去重）
# 反向映射：H1类别ID→晶系名称（用于后续标签转名称）
h1_to_crystal = {v: k for k, v in crystal_to_h1.items()}
# 获取所有唯一晶系（按H1类别ID排序）
all_crystal_systems = [h1_to_crystal[h1] for h1 in sorted(crystal_to_h1.values())]
print(f"数据集包含的晶系列表: {all_crystal_systems}")

# 2. 加载空间群-材料ID映射，生成测试集划分（复用训练逻辑）
spacegroup_to_materials = load_materials_by_spacegroup(
)

# 3. 生成测试集划分（仅保留测试集，与训练时种子/比例一致）
def split_by_spacegroup_test(spacegroup_to_materials, root_dir, num_views=2, seed=42):
    split_dict = {}
    np.random.seed(seed)
    for sg, materials in spacegroup_to_materials.items():
        for mid in materials:
            material_dir = os.path.join(root_dir, str(sg), mid)
            if not os.path.isdir(material_dir):
                continue
            views = sorted([f for f in os.listdir(material_dir) if f.endswith('.png')])
            if len(views) < num_views:
                continue
            # 枚举所有视图组合
            from itertools import combinations
            all_combs = list(combinations(views, num_views))
            # 复用训练时的测试集比例（0.15）
            total = len(all_combs)
            test_split = int(total * 0.15)
            for i, comb in enumerate(all_combs):
                key = (mid, comb)
                if i < test_split:
                    split_dict[key] = 'test'
    return split_dict

split_dict = split_by_spacegroup_test(
    spacegroup_to_materials,
    args.root_dir,
    num_views=args.num_views,
    seed=12  # 与训练时一致，保证测试集完全相同
)
print(f"测试集总样本数: {len([k for k, v in split_dict.items() if v == 'test'])}")

# ====================== 测试集数据加载 ======================
# 1. 预处理（与训练时一致，仅标准化）
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

# 2. 多视图collate_fn（与训练时一致）
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
        torch.stack(imgs_list),
        paths_list
    )

# 3. 创建测试集DataLoader
test_dataset = MultiviewImgDataset(
    root_dir=args.root_dir,
    split='test',
    split_dict=split_dict,
    num_views=args.num_views,
    transform=transform_test
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batchSize,
    shuffle=False,
    num_workers=4,
    drop_last=False,
    pin_memory=True,
    collate_fn=mv_collate_fn
)
print(f"实际加载的测试集样本数: {len(test_dataset)}")

# ====================== 加载模型 ======================
# 1. 加载SVBCNN骨干（与训练时一致）
svBCNN = SVBCNN(args.BCNN_name, pretraining=False, BCNN_name=args.BCNN_name)
# 2. 加载MVBCNN模型
mvBCNN = MVBCNN(args.BCNN_name, svBCNN, BCNN_name=args.BCNN_name, num_views=args.num_views)
# 3. 加载训练权重
model_path = '/internfs/pengqianwen/MVBBCNN/results/3/22230c3c-06fc-4bb7-928b-5e171867397f/avg/3/model-00004.pth'
checkpoint = torch.load(model_path, map_location=device)
mvBCNN.load_state_dict(checkpoint, strict=False)  # 忽略无关参数（如并行训练标识）
mvBCNN.to(device)
mvBCNN.eval()  # 切换评估模式
print(f"成功加载模型: {model_path}")

# ====================== 核心功能：按晶系拆分评估 ======================
def evaluate_by_crystal_system(model, test_loader, device, save_dir, crystal_save_dir):
    # 1. 存储全量结果（含晶系信息）
    results = defaultdict(list)  # key: 晶系名称, value: [真实H1, 预测H1, 真实H2, 预测H2, 材料ID]
    
    # 2. 模型推理（禁用梯度计算）
    with torch.no_grad():
        for batch_idx, (class_id_h1, class_id_h2, imgs, paths) in enumerate(test_loader):
            # 数据移至设备
            imgs = imgs.to(device)
            class_id_h1 = class_id_h1.to(device)
            class_id_h2 = class_id_h2.to(device)
            
            # 模型输出（H1: 晶系分类，H2: 空间群分类）
            outputs_h1, outputs_h2 = model(imgs)
            pred_h1 = torch.argmax(outputs_h1, dim=1)
            pred_h2 = torch.argmax(outputs_h2, dim=1)
            
            # 解析材料ID（从path中提取，需匹配你的文件路径格式）
            # 假设path格式：".../空间群ID/材料ID/视图.png"，则材料ID是path.split('/')[-2]
            batch_material_ids = [path.split('/')[-2] for path in paths]
            
            # 将结果按晶系分组
            for idx in range(len(class_id_h1)):
                # 从材料ID获取晶系名称
                mid = batch_material_ids[idx]
                crystal = material_to_crystal.get(mid, "未知晶系")  # 兜底：避免材料ID未匹配
                # 保存当前样本结果（转CPU→numpy）
                results[crystal].append([
                    class_id_h1[idx].cpu().item(),  # 真实H1
                    pred_h1[idx].cpu().item(),      # 预测H1
                    class_id_h2[idx].cpu().item(),  # 真实H2
                    pred_h2[idx].cpu().item(),      # 预测H2
                    mid                             # 材料ID
                ])
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                print(f"处理批次 {batch_idx + 1}/{len(test_loader)} | 已收集 {sum(len(v) for v in results.values())} 个样本")
    
    # 3. 全量结果统计（先计算整体性能，再拆分晶系）
    all_true_h1 = []
    all_pred_h1 = []
    all_true_h2 = []
    all_pred_h2 = []
    crystal_perf = {}  # 存储每个晶系的性能指标
    
    for crystal, samples in results.items():
        # 提取当前晶系的所有样本数据
        true_h1 = [s[0] for s in samples]
        pred_h1 = [s[1] for s in samples]
        true_h2 = [s[2] for s in samples]
        pred_h2 = [s[3] for s in samples]
        mids = [s[4] for s in samples]
        
        # 累加全量数据
        all_true_h1.extend(true_h1)
        all_pred_h1.extend(pred_h1)
        all_true_h2.extend(true_h2)
        all_pred_h2.extend(pred_h2)
        
        # 计算当前晶系的性能指标
        # 3.1 H1（晶系内）准确率：当前晶系样本中，H1预测正确的比例
        h1_acc = accuracy_score(true_h1, pred_h1)
        # 3.2 H2（空间群）准确率：当前晶系样本中，H2预测正确的比例
        h2_acc = accuracy_score(true_h2, pred_h2)
        # 3.3 分类报告（精确率/召回率/F1）
        h1_report = classification_report(true_h1, pred_h1, output_dict=True, zero_division=0)
        h2_report = classification_report(true_h2, pred_h2, output_dict=True, zero_division=0)
        # 3.4 样本数量
        sample_count = len(samples)
        
        # 保存当前晶系的性能
        crystal_perf[crystal] = {
            "sample_count": sample_count,
            "h1_accuracy": h1_acc,
            "h2_accuracy": h2_acc,
            "h1_classification_report": h1_report,
            "h2_classification_report": h2_report,
            "samples": samples  # 原始样本明细（含材料ID）
        }
        
        # 打印当前晶系的关键性能
        print(f"\n【{crystal}】")
        print(f"样本数: {sample_count} | H1（晶系）准确率: {h1_acc:.4f} | H2（空间群）准确率: {h2_acc:.4f}")

    # 4. 整体性能统计
    overall_perf = {
        "total_samples": len(all_true_h1),
        "overall_h1_accuracy": accuracy_score(all_true_h1, all_pred_h1),
        "overall_h2_accuracy": accuracy_score(all_true_h2, all_pred_h2),
        "all_crystal_systems": list(crystal_perf.keys())
    }

    # 5. 结果保存（分3类：整体报告、晶系明细、可视化）
    # 5.1 保存整体性能报告（JSON）
    with open(os.path.join(save_dir, "overall_evaluation.json"), 'w', encoding='utf-8') as f:
        json.dump({
            "overall_perf": overall_perf,
            "crystal_system_summary": {k: {
                "sample_count": v["sample_count"],
                "h1_accuracy": v["h1_accuracy"],
                "h2_accuracy": v["h2_accuracy"]
            } for k, v in crystal_perf.items()}
        }, f, indent=4, ensure_ascii=False)

    # 5.2 保存每个晶系的详细报告（CSV+JSON）
    for crystal, perf in crystal_perf.items():
        # 保存样本明细（CSV）
        detail_df = pd.DataFrame(
            perf["samples"],
            columns=["true_h1", "pred_h1", "true_h2", "pred_h2", "material_id"]
        )
        # 添加晶系名称列，方便后续合并分析
        detail_df["crystal_system"] = crystal
        # 保存CSV（晶系名称替换特殊字符，避免路径错误）
        safe_crystal_name = crystal.replace("/", "_").replace(" ", "_")
        detail_df.to_csv(
            os.path.join(crystal_save_dir, f"{safe_crystal_name}_details.csv"),
            index=False,
            encoding='utf-8'
        )
        
        # 保存性能报告（JSON）
        with open(os.path.join(crystal_save_dir, f"{safe_crystal_name}_perf.json"), 'w', encoding='utf-8') as f:
            json.dump(perf, f, indent=4, ensure_ascii=False)

    # 6. 可视化（分晶系展示关键指标）
    # 6.1 各晶系样本数与H1准确率对比图
    crystal_names = list(crystal_perf.keys())
    sample_counts = [crystal_perf[c]["sample_count"] for c in crystal_names]
    h1_accs = [crystal_perf[c]["h1_accuracy"] for c in crystal_names]
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    # 左轴：样本数（柱状图）
    ax1.bar(crystal_names, sample_counts, color='#1f77b4', alpha=0.6, label='样本数')
    ax1.set_xlabel('晶系', fontsize=12)
    ax1.set_ylabel('样本数量', color='#1f77b4', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    # 右轴：H1准确率（折线图）
    ax2 = ax1.twinx()
    ax2.plot(crystal_names, h1_accs, color='#ff7f0e', marker='o', linewidth=2, label='H1准确率')
    ax2.set_ylabel('H1（晶系）准确率', color='#ff7f0e', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax2.set_ylim(0, 1.05)  # 准确率范围固定0-1.05，便于对比
    # 添加标题和图例
    plt.title(f'各晶系样本数与H1准确率对比\n整体H1准确率: {overall_perf["overall_h1_accuracy"]:.4f}', fontsize=14)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, "crystal_system_h1_acc.png"), dpi=300)
    plt.close()

    # 6.2 各晶系H2准确率对比（柱状图）
    h2_accs = [crystal_perf[c]["h2_accuracy"] for c in crystal_names]
    plt.figure(figsize=(14, 8))
    bars = plt.bar(crystal_names, h2_accs, color='#2ca02c', alpha=0.7)
    # 在柱子上添加数值标签
    for bar, acc in zip(bars, h2_accs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{acc:.4f}', ha='center', va='bottom', fontsize=10)
    plt.xlabel('晶系', fontsize=12)
    plt.ylabel('H2（空间群）准确率', fontsize=12)
    plt.title(f'各晶系H2准确率对比\n整体H2准确率: {overall_perf["overall_h2_accuracy"]:.4f}', fontsize=14)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45)  # 晶系名称过长时旋转标签
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "crystal_system_h2_acc.png"), dpi=300)
    plt.close()

    # 6.3 整体H1混淆矩阵（标注晶系名称）
    plt.figure(figsize=(12, 10))
    cm_h1 = confusion_matrix(all_true_h1, all_pred_h1)
    # 用晶系名称替换类别ID作为标签
    crystal_labels = [h1_to_crystal.get(h1, f'未知_{h1}') for h1 in sorted(set(all_true_h1))]
    sns.heatmap(
        cm_h1,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=crystal_labels,
        yticklabels=crystal_labels
    )
    plt.title(f'整体H1（晶系）混淆矩阵\n准确率: {overall_perf["overall_h1_accuracy"]:.4f}', fontsize=14)
    plt.xlabel('预测晶系', fontsize=12)
    plt.ylabel('真实晶系', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "overall_h1_confusion_matrix.png"), dpi=300)
    plt.close()

    # 7. 打印最终总结
    print("\n" + "="*80)
    print("按晶系拆分评估总结")
    print("="*80)
    print(f"整体测试集样本数: {overall_perf['total_samples']}")
    print(f"整体H1（晶系）准确率: {overall_perf['overall_h1_accuracy']:.4f}")
    print(f"整体H2（空间群）准确率: {overall_perf['overall_h2_accuracy']:.4f}")
    print("\n各晶系关键指标：")
    print(f"{'晶系':<15} {'样本数':<8} {'H1准确率':<12} {'H2准确率':<12}")
    print("-"*50)
    for crystal in crystal_names:
        perf = crystal_perf[crystal]
        print(f"{crystal:<15} {perf['sample_count']:<8} {perf['h1_accuracy']:<12.4f} {perf['h2_accuracy']:<12.4f}")
    print("\n结果文件保存路径:")
    print(f"- 整体报告: {os.path.join(save_dir, 'overall_evaluation.json')}")
    print(f"- 晶系明细: {crystal_save_dir}")
    print(f"- 可视化图表: {save_dir}")

    return overall_perf, crystal_perf

# ====================== 执行评估 ======================
if __name__ == "__main__":
    evaluate_by_crystal_system(
        model=mvBCNN,
        test_loader=test_loader,
        device=device,
        save_dir=args.save_dir,
        crystal_save_dir=crystal_system_save_dir
    )
    print(f"\n评估完成！所有结果已保存至: {args.save_dir}")

