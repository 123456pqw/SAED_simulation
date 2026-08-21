import numpy as np
import torch
import glob
import json
import torch.nn as nn
import seaborn as sns
from skimage import io, transform
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix
from models.MVBCNN import MVBCNN, SVBCNN
import torch.nn.functional as F
from torchvision import transforms, datasets
import os
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

def plot_crystal_system_confidence(probs, class_names, save_path="crystal_confidence.svg"):
    """
    绘制晶系分类置信度柱状图（SVG可编辑字体）
    
    参数：
    probs : numpy array (num_classes,)
    class_names : list
    save_path : 输出SVG路径
    """

    # ===== 排序（从高到低）=====
    sorted_idx = np.argsort(-probs)
    probs_sorted = probs[sorted_idx]
    labels_sorted = [class_names[i] for i in sorted_idx]

    # ===== 设置全局字体（关键！SVG可编辑）=====
    plt.rcParams['svg.fonttype'] = 'none'   # ⭐ 核心：不转路径
    plt.rcParams['font.family'] = 'Arial'   # 或 'Times New Roman'

    # ===== 创建图像 =====
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    bars = ax.bar(labels_sorted, probs_sorted)

    # ===== 美化 =====
    ax.set_ylabel("Confidence", fontsize=12)
    ax.set_xlabel("Crystal System", fontsize=12)
    ax.set_title("Crystal System Prediction Confidence", fontsize=14)

    ax.set_ylim(0, 1)

    # 数值标注
    for bar, val in zip(bars, probs_sorted):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + 0.02,
                f"{val:.2f}",
                ha='center', va='bottom', fontsize=10)

    # 旋转标签（防止挤）
    plt.xticks(rotation=15)

    plt.tight_layout()

    # ===== 保存SVG =====
    plt.savefig(save_path, format='svg')
    plt.close()

    print(f"Saved SVG to {save_path}")

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="mvBCNN_neww")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=8)
parser.add_argument("-num_two", type=int, help="number of models per class", default=1)
parser.add_argument("-num_val", type=int, help="number of val ", default=2)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0001)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-BCNN_name", "--BCNN_name", type=str, help="BCNN model name", default="resnet18")
parser.add_argument("-num_views", type=int, help="number of views", default=2)
parser.add_argument("-test_mode",  action='store_true',help="run or not")#不加上就是使用数据增强
parser.add_argument("-train_path", type=str, default="")
parser.add_argument("-val_path", type=str, default="")
parser.set_defaults(train=False)

from PIL import Image
import torch
import os
from torchvision import transforms


def visualize_confidence(h2_out, class_names, top_k=10, figsize=(14, 14), dpi=300):
    """
    改进版置信度可视化函数
    
    参数：
    h2_out : 模型输出logits张量
    top_k : 显示的主要类别数量
    figsize : 图像尺寸
    dpi : 分辨率
    """
    # 转换为概率并取第一个样本（假设batch_size=1）
    with torch.no_grad():  # 禁用梯度计算
        probs = torch.softmax(h2_out.detach().cpu(), dim=1).numpy()[0]  # 正确分离顺序
    
    
    # 获取排序后的索引和概率
    sorted_indices = np.argsort(-probs)
    sorted_probs = probs[sorted_indices]
    
    # 合并小概率类别
    main_probs = sorted_probs[:top_k]
    other_prob = sorted_probs[top_k:].sum()
    values = np.append(main_probs, other_prob)
    # 使用真实类别名称
    labels = [class_names[i] for i in sorted_indices[:top_k]] + ['Others']

    # 生成颜色时增加OTHERS专用颜色
    base_cmap = get_cmap('Spectral')  # 改用更鲜艳的色系
    colors = base_cmap(np.linspace(0.1, 0.9, top_k))  # 避免使用色系两端过浅的颜色
    colors = np.vstack([colors, [0.5, 0.5, 0.5, 1]])  # 调整OTHERS灰色饱和度

    
    # 创建极坐标玫瑰图
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='#f5f5f5')
    ax = fig.add_subplot(111, polar=True)
    
    # 计算角度参数
    theta = np.linspace(0.0, 2 * np.pi, len(values), endpoint=False)
    width = 2 * np.pi / len(values) * 0.85
    
    # 绘制带渐变的扇形
    bars = ax.bar(theta, values, width=width,
                 edgecolor='black',  # 增强边界对比度
                 linewidth=1.2,
                 color=colors,        # 确保直接使用颜色数组
                 alpha=0.95) 
    
    # 添加3D效果
    for bar, height in zip(bars, values):
        bar.set_alpha(0.9)
        bar.set_zorder(3)
        bar.set_edgecolor((0.2, 0.2, 0.2, 0.5))
        bar.set_hatch('..' if height < 0.05 else None)
    
    # 添加环形注释
    ax.annotate(f"Top-{top_k} Confidence", xy=(0.5, 0.5), xytext=(0, 0),
                xycoords='axes fraction', textcoords='offset points',
                ha='center', va='center', fontsize=16,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", lw=1))
    
    # 设置极坐标轴
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.grid(color='lightgrey', linestyle='--', linewidth=0.7)
    
    # 创建图例
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                label=labels[i], 
                                markerfacecolor=colors[i], 
                                markersize=15) for i in range(len(labels))]
    
    ax.legend(handles=legend_elements, loc='upper right',
             bbox_to_anchor=(1.25, 1.1), frameon=False,
             fontsize=10, title="Class Labels",
             title_fontsize=12)
    
    # 保存高清图像
    plt.savefig('confidence_visualization.png', 
               bbox_inches='tight', 
               pad_inches=0.5,
               transparent=False)
    plt.close()

def threshold_brightness(image, threshold=50):
    """
    将亮度值低于指定阈值的像素设置为黑色。
    
    :param image: 输入的PIL图像。
    :param threshold: 亮度阈值，低于此值的像素将被设置为黑色。
    :return: 处理后的图像。
    """
    # 将图像转换为灰度图
    image_gray = image.convert('L')
    
    # 将图像转为NumPy数组
    image_np = np.array(image_gray)
    
    # 创建一个与图像大小相同的掩膜，标记低于阈值的像素
    mask = image_np < threshold
    
    # 将低于阈值的像素设置为0（黑色）
    image_np[mask] = 0
    
    # 将处理后的数组转换为图像
    processed_image = Image.fromarray(image_np)
    
    return processed_image


# 中值滤波去噪
def remove_median_noise(image, kernel_size=5):
    image_np = np.array(image)
    denoised_image_np = cv2.medianBlur(image_np, kernel_size)
    return Image.fromarray(denoised_image_np)


def pad_image_to_target_size(image, target_size=(720, 720)):
    """
    Resize the image while maintaining the aspect ratio, then pad the image to the target size.
    
    参数:
    - image: 输入图像 (PIL)
    - target_size: 目标图像尺寸 (width, height)
    
    返回:
    - 返回一个填充后的图像
    """
    # Get the original size
    width, height = image.size
    
    # Calculate aspect ratio
    aspect_ratio = width / height
    
    # Resize the image to maintain the aspect ratio
    if width > height:
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)
    
    # Resize the image
    image = image.resize((new_width, new_height))

    # Create a new blank image with the target size
    padded_image = Image.new('RGB', target_size, (255, 255, 255))  # white padding
    # Paste the resized image onto the center of the blank image
    padded_image.paste(image, ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2))

    return padded_image


def ImageDataset_mul(image_paths, transform, target_size=(720, 720), save_dir='processed_images'):
    """
    处理和转换多张图片，确保它们有相同的大小，并将它们堆叠成一个单一的张量，同时保存每个处理后的图像。

    参数:
    - image_paths: 图像路径列表
    - transform: 图像转换方法
    - target_size: 目标图像尺寸，确保所有图像大小一致 (width, height)
    - save_dir: 保存处理后图像的文件夹路径
    
    返回:
    - 返回一个堆叠的图像张量
    """
    imgs = []
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 创建文件夹保存处理后的图像

    for i, path in enumerate(image_paths):
        img = Image.open(path)
        # Resize and pad the image to the target size
        #img = pad_image_to_target_size(img, target_size)
        # 获取图像的尺寸
        w, h = img.size

        # 计算裁剪框的宽度和高度
        crop_ratio=0.6
        crop_width = int(w * crop_ratio)
        crop_height = int(h * crop_ratio)

        # 计算裁剪框的左上角和右下角坐标
        left = (w - crop_width) // 2
        upper = (h - crop_height) // 2
        right = left + crop_width
        lower = upper + crop_height

        # 裁剪图像
        img = img.crop((left, upper, right, lower))
        img = remove_median_noise(img, kernel_size=1)
        img = remove_median_noise(img, kernel_size=3)
        if img.mode != 'RGB':
            img = img.convert('RGB')  # Ensure the image has 3 channels
        target_size = (512, 512)  # 你可以根据需要调整目标尺寸
        target_size = (1024, 1024)
        img = img.resize(target_size, Image.BICUBIC)
        # Save the processed image
        save_path = os.path.join(save_dir, f'processed_{i+1}.jpg')  # 保存的文件名可以根据需求命名
        img.save(save_path)
        print(f"Saved processed image to: {save_path}")

        # Apply the transformation (e.g., normalization)
        if transform:
            img = transform(img)

        imgs.append(img)

    # 将所有图像堆叠成一个单一的张量
    return torch.stack(imgs)

def ImageDataset_single(path, transform):
    im = Image.open(path).convert('RGB')
    crop_box = (79, 99, 471, 471)  # Crop region (customize as needed)
    #im = im.crop(crop_box)
    if transform:
        im = transform(im)
    return im.unsqueeze(0)

def generate_heatmap(model, input_tensor, original_img_path, save_path, target_layer_name='layer4'):
    """
    生成Grad-CAM热力图并叠加到原图
    参数：
    model : 模型实例
    input_tensor : 输入张量 (C,H,W)
    original_img_path : 原始图像路径
    save_path : 保存路径
    target_layer_name : 目标层名称
    """
    # 获取目标层
    target_layer = None
    for name, module in model.named_modules():
        if name == target_layer_name:
            target_layer = module
            break
    
    if target_layer is None:
        raise ValueError(f"未找到指定层：{target_layer_name}")

    # 注册钩子
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output.detach())
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
    
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    # 前向传播
    model.zero_grad()
    outputs = model(input_tensor.unsqueeze(0))
    class_idx = torch.argmax(outputs).item()
    
    # 反向传播
    outputs[:, class_idx].backward()

    # 计算CAM
    activation = activations[0]
    gradient = gradients[0]
    
    weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activation, dim=1, keepdim=True)
    
    # 后处理
    cam = F.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = F.interpolate(cam, input_tensor.shape[1:], mode='bilinear', align_corners=False)[0,0].cpu().numpy()

    # 叠加到原图
    original_img = Image.open(original_img_path).convert('RGB')
    heatmap_img = (plt.cm.jet(cam)[..., :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_img).resize(original_img.size)
    
    blended = Image.blend(original_img, heatmap_img, alpha=0.5)
    
    # 保存结果
    blended.save(save_path)
    
    # 移除钩子
    forward_handle.remove()
    backward_handle.remove()


