import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import shutil
import matplotlib
shutil.rmtree(matplotlib.get_cachedir())
import os
import torch.nn.functional as F
import seaborn as sns
import shutil
import json
import argparse
import uuid
import pandas as pd
from sklearn.metrics import accuracy_score  # 需要先导入
from sklearn.metrics import confusion_matrix
from tools.ImgDataset_random_aug import MultiviewImgDataset, SingleImgDataset
from models.MVBCNN import MVBCNN, SVBCNN
from models.mix import MVBCNN_Comparison
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.font_manager as fm
from matplotlib import rcParams, font_manager
from utils.simulate import generate_beam_directions, load_materials_by_spacegroup, extract_spacegroup_number
import random
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42   # PDF: 保留 TrueType 字体
mpl.rcParams['ps.fonttype'] = 42    # PS: 保留 TrueType 字体
mpl.rcParams['svg.fonttype'] = 'none'  # SVG: 保留文字


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

    return (
        torch.tensor(class_id_h1_list),
        torch.tensor(class_id_h2_list),
        torch.stack(imgs_list),  # final shape: [B, V, C, H, W]
        paths_list
    )
def split_by_spacegroup(spacegroup_to_materials, root_dir, num_views=2, seed=42, 
                        val_ratio=0.15, test_ratio=0.15, max_samples_per_group=5, max_combs_per_material=20):
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

            # 限制每个材料最多取多少个组合
            if max_combs_per_material is not None and len(all_combs) > max_combs_per_material:
                all_combs = random.sample(all_combs, max_combs_per_material)

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

plt.rcParams.update({
    'font.size': 6,          # 基础字体大小
    'axes.titlesize': 7,     # 标题字体大小
    'axes.labelsize': 6,     # 坐标轴标签字体
    'xtick.labelsize': 6,     # X轴刻度字体
    'ytick.labelsize': 6,     # Y轴刻度字体
    'legend.fontsize': 5      # 图例字体
})

def save_metrics_to_csv(classnames, metrics, save_path):
    report = {
        'Class': classnames,
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1': metrics['f1'],
        'ACC':metrics['accuracy'],
        'AUC': [metrics['roc_auc'][i] for i in range(len(classnames))]
    }
    
    # 添加宏观平均
    report['Class'].append('Macro-average')
    report['Precision'] = np.append(report['Precision'], np.mean(metrics['precision']))
    report['Recall'] = np.append(report['Recall'], np.mean(metrics['recall']))
    report['F1'] = np.append(report['F1'], np.mean(metrics['f1']))
    report['AUC'] = np.append(report['AUC'], metrics['roc_auc']['macro'])
    
    pd.DataFrame(report).to_csv(os.path.join(save_path, 'classification_report.csv'), index=False)

def plot_roc_curves(metrics, class_names, save_path):
    plt.figure(figsize=(3.15, 3), dpi=300)
    colors = [
        '#4E79A7', '#F28E2B', '#E15759', '#76B7B2',
        '#59A14F', '#EDC948', '#B07AA1'
    ]
    
    
    # 绘制曲线
    for i, color in zip(range(len(class_names)), colors):
        plt.plot(metrics['fpr'][i], metrics['tpr'][i], 
                color=color, lw=2,
                label=f'{class_names[i]} (AUC = {metrics["roc_auc"][i]:.2f})')

    # 添加宏平均曲线
    plt.plot(metrics['fpr']["macro"], metrics['tpr']["macro"],
            label=f'Macro-average (AUC = {metrics["roc_auc"]["macro"]:.2f})',
            color='navy', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # 坐标轴标签设置
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    
    # 修正图例参数
    plt.legend(loc="lower right")  # 关键修改点
    
    plt.savefig(os.path.join(save_path, 'roc_curve.png'))
    plt.savefig(os.path.join(save_path, 'roc_curve.pdf'), bbox_inches='tight')  # PDF 矢量，可编辑文字
    plt.savefig(os.path.join(save_path, 'roc_curve.svg'), bbox_inches='tight')  # SVG 矢量，可编辑文字

    plt.close()

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="mvBCNN_neww")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=8)
parser.add_argument("-num_two", type=int, help="number of models per class", default=8)
parser.add_argument("-num_val", type=int, help="number of val ", default=2000000)
parser.add_argument("-rate1", type=float, help="number of val ", default=0.9)
parser.add_argument("-rate2", type=float, help="number of val ", default=0.2)
parser.add_argument("-nstop", type=int, help="number of val ", default=5)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0001)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-BCNN_name", "--BCNN_name", type=str, help="BCNN model name", default="resnet18")
parser.add_argument("-num_views", type=int, help="number of views", default=2)
parser.add_argument("-test_mode",  action='store_true',help="run or not")#不加上就是使用数据增强
parser.add_argument("-train_path", type=str, default="")
parser.add_argument("-val_path", type=str, default="")
parser.add_argument("-root_dir", type=str, default="")
parser.set_defaults(train=False)



def create_folder(log_dir):
    """Create a folder for logging. If it exists, delete and recreate it."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    else:
        print('WARNING: Summary folder already exists! It will be overwritten!')
        shutil.rmtree(log_dir)
        os.makedirs(log_dir)

def calculate_metrics(y_true, y_pred, y_prob, class_names):
    

    # 基础指标计算
    num_classes = len(class_names)
    print("all_probs.shape:", y_prob.shape)

    #assert y_prob.shape[1] == num_classes, "预测概率维度与类别数量不匹配"

    # 新增准确率计算
    accuracy = accuracy_score(y_true, y_pred)
    
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    # 多分类ROC AUC计算（使用One-vs-Rest策略）
    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
    fpr, tpr, roc_auc = {}, {}, {}

    print("y_true unique:", np.unique(y_true))
    print("y_true_bin.shape:", y_true_bin.shape)
    print("y_prob.shape:", y_prob.shape)

    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算宏观平均AUC
    fpr["macro"], tpr["macro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return {
        'accuracy': accuracy,  # 新增返回项
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr
    }

# Custom JSON Encoder to handle numpy.float32
def numpy_to_float32(obj):
    if isinstance(obj, np.float32):
        return float(obj)  # Convert numpy.float32 to Python float
    raise TypeError(f"Type {obj.__class__.__name__} not serializable")


def plot_confusion_matrix(cm, classnames, save_path=None):
    plt.figure(figsize=(3.15, 3.15), dpi=300)
    
    # 计算行归一化比例（按真实类别）
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # 处理除零情况
    # 创建渐变色板
    custom_cmap = sns.light_palette("#4B0082", as_cmap=True)
    
    # 绘制热力图（显示比例）
    ax = sns.heatmap(
        cm_norm, 
        annot=True,  # 启用注释
        fmt=".2f",   # 显示两位小数
        cmap=custom_cmap,
        square=True,
        linewidths=0.5,
        cbar_kws={
            'shrink': 0.6,
            'label': 'Proportion'  # 修改颜色条标签
        },
        xticklabels=classnames,
        yticklabels=classnames
    )
    
    # 调整坐标轴标签
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha='right'
    )
    
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0
    )
    
    # 添加标签和标题
    plt.xlabel('Predicted', fontsize=5)
    plt.ylabel('True', fontsize=5)
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.03)

         # 同时保存一份 SVG
        base, _ = os.path.splitext(save_path)
        svg_path = base + ".svg"
        plt.savefig(svg_path, format="svg", bbox_inches='tight', pad_inches=0.03)

    plt.close()


# 评估模型并计算混淆矩阵
def evaluate_model(model, val_loader, classnames1,save_dir=None):
    all_preds_1 = []
    all_probs = []
    all_targets_1 = []

    incorrect_samples_per_class1 = {classname: [] for classname in classnames1}  # 存储每个类分类错误的样本信息

    # 评估模式下进行推理
    model.eval()
    with torch.no_grad():
        for data in val_loader:
            x = data[2].cuda()
            B, V, C, H, W = x.shape
            x = x.view(B*V, C, H, W)
            target_h1 = data[0].cuda().long()  # Ground truth targets
            target_h2 = data[1].cuda().long()
            # 获取模型预测
            h1_out, h2_out = model(x)
            h1_prob = F.softmax(h1_out, dim=1)
            
            all_probs.extend(h1_prob.cpu().numpy())
            pred_h1 = torch.max(h1_out, 1)[1]
            pred_h2 = torch.max(h2_out, 1)[1]

            # 将预测和真实标签保存下来
            all_preds_1.extend(pred_h1.cpu().numpy())
            all_targets_1.extend(target_h1.cpu().numpy())

            
    # 计算混淆矩阵
    cm1 = confusion_matrix(all_targets_1, all_preds_1, labels=list(range(len(classnames1))))

    
    metrics = calculate_metrics(all_targets_1, all_preds_1, np.array(all_probs), classnames1)
    



    # 打印总体准确率
    overall_accuracy_1 = np.sum(np.diag(cm1)) / np.sum(cm1)
    print(f'Overall Accuracy: {overall_accuracy_1 * 100:.2f}%')


    # 保存混淆矩阵和准确率柱状图
    if save_dir:
        cm_save_path = os.path.join(save_dir, 'confusion_matrix_h1.png')
        plot_confusion_matrix(cm1, classnames1, cm_save_path)
        save_metrics_to_csv(classnames1, metrics, save_dir)
        plot_roc_curves(metrics, classnames1, save_dir)
    
    


def evaluate_model_single(model, val_loader, classnames1,save_dir=None):
    all_preds_1 = []
    all_probs = []
    all_targets_1 = []

    incorrect_samples_per_class1 = {classname: [] for classname in classnames1}  # 存储每个类分类错误的样本信息

    # 评估模式下进行推理
    model.eval()
    with torch.no_grad():
        for data in val_loader:
            x = data[2].cuda()
            target_h1 = data[0].cuda().long()  # Ground truth targets
            target_h2 = data[1].cuda().long()
            # 获取模型预测
            h1_out, h2_out = model(x)
            h1_prob = F.softmax(h1_out, dim=1)
            
            all_probs.extend(h1_prob.cpu().numpy())
            pred_h1 = torch.max(h1_out, 1)[1]
            pred_h2 = torch.max(h2_out, 1)[1]

            # 将预测和真实标签保存下来
            all_preds_1.extend(pred_h1.cpu().numpy())
            all_targets_1.extend(target_h1.cpu().numpy())

            
    # 计算混淆矩阵
    cm1 = confusion_matrix(all_targets_1, all_preds_1, labels=list(range(len(classnames1))))


    metrics = calculate_metrics(all_targets_1, all_preds_1, np.array(all_probs), classnames1)
    
    # 打印详细指标
    print("\nDetailed Classification Report:")
    for i, name in enumerate(classnames1):
        print(f"{name}:")
        print(f"  Precision: {metrics['precision'][i]:.4f}")
        print(f"  Recall:    {metrics['recall'][i]:.4f}") 
        print(f"  F1-Score:  {metrics['f1'][i]:.4f}")
        print(f"  AUC:       {metrics['roc_auc'][i]:.4f}\n")
    


    # 打印总体准确率
    overall_accuracy_1 = np.sum(np.diag(cm1)) / np.sum(cm1)
    print(f'Overall Accuracy: {overall_accuracy_1 * 100:.2f}%')


    # 保存混淆矩阵和准确率柱状图
    if save_dir:
        cm_save_path = os.path.join(save_dir, 'confusion_matrix_h1.png')
        plot_confusion_matrix(cm1, classnames1, cm_save_path)
        save_metrics_to_csv(classnames1, metrics, save_dir)
        plot_roc_curves(metrics, classnames1, save_dir)
    
   

