import numpy as np
import os
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from .Model import Model 
import re
from torch.nn.parallel import DataParallel, gather
from torch.nn.parameter import Parameter

# 预处理参数（保持与ImageNet一致）
mean = torch.tensor([0.485, 0.456, 0.406], device='cuda')
std = torch.tensor([0.229, 0.224, 0.225], device='cuda')

csv_file = '/internfs/pengqianwen/MVBBCNN/utils/data.csv'  # 替换为您的 CSV 文件路径
data = pd.read_csv(csv_file)  # 跳过第一行
classnames = data.iloc[:, 0].tolist()

# ResNet18 基础模块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

# 单视角网络
class SVBCNN(Model):
    def __init__(self, name, nclasses=7, pretraining=True, BCNN_name='resnet18'):
        super(SVBCNN, self).__init__(name)
        self.log_var_h1 = nn.Parameter(torch.tensor(1.0))  # log(σ^2)
        self.log_var_h2 = nn.Parameter(torch.tensor(0.0))

        # ResNet18 主干网络
        resnet = models.resnet18(pretrained=pretraining)
        self.features = nn.Sequential(
            *list(resnet.children())[:-2]  # 输出尺寸 [batch, 512, 7, 7]
        )

        # 任务头配置
        self.task1_head = self._build_task_head(512, 7)  # 晶体系统分类
        self.task2_head = self._build_task_head(256, len(classnames))  # 空间群分类
        
        # 任务2特征增强
        self.task2_feature = nn.Sequential(
            BasicBlock(512, 256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.MaxPool2d(2)
        )

    def _build_task_head(self, in_dim, out_dim):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_dim, out_dim)
        )

    def extract_feature(self, x, mode='task2', with_normalize=True):

        base_feat = self.features(x)  # [batch, 512, 7, 7]
        
        # 多模式特征提取
        if mode == 'base':
            features = base_feat
        elif mode == 'task1':
            features = self.task1_head[0](base_feat)  # AdaptiveAvgPool2d
            features = self.task1_head[1](features)   # Flatten [batch, 512]
        elif mode == 'task2':
            task2_feat = self.task2_feature(base_feat)  # [batch, 256, 3, 3]
            features = self.task2_head[0](task2_feat)   # AdaptiveAvgPool2d
            features = self.task2_head[1](features)     # Flatten [batch, 256]
        else:
            raise ValueError("Invalid mode, choose from ['base', 'task1', 'task2']")
        
        # L2归一化（推荐用于相似度计算）
        if with_normalize and features.ndim > 2:
            features = F.normalize(features.flatten(1), p=2, dim=1)
        elif with_normalize:
            features = F.normalize(features, p=2, dim=1)
            
        return features

    def forward(self, x):
        base_feat = self.features(x)  # [batch, 512, 7, 7]
        
        # 任务1处理
        y1 = self.task1_head(base_feat)
        
        # 任务2处理
        task2_feat = self.task2_feature(base_feat)  # [batch, 256, 3, 3]
        y2 = self.task2_head(task2_feat)
        
        return y1, y2

# 多视角网络
class MVBCNN(Model):
    def __init__(self, name, svBCNN, nclasses_h1=7, nclasses_h2=222, BCNN_name='resnet18', num_views=2):
        super(MVBCNN, self).__init__(name)
        
        # 共享组件
        self.features = svBCNN.features
        self.task2_feature = svBCNN.task2_feature
        self.num_views = num_views
        self.log_var_h1 = svBCNN.log_var_h1  # log(σ^2)
        self.log_var_h2 = svBCNN.log_var_h2

        self.features = svBCNN.features
        for param in self.features.parameters():  # 冻结ResNet主干
            param.requires_grad = True

        self.task2_feature = svBCNN.task2_feature
        for param in self.task2_feature.parameters():  # 冻结任务2特征增强层
            param.requires_grad = True

        # 多视角融合模块
        self.view_attention = ViewAttention(512)  # 输入通道调整为512
        self.view_pooling = HybridPooling()
        
        # 任务头
        self.task1_head = svBCNN.task1_head
        self.task2_head = svBCNN.task2_head
    
    def extract_feature(self, x, mode='task2', with_normalize=False):
        #print(">>> no_grad mode:", torch.is_grad_enabled())

        """
        提取 MVBCNN 的中间特征，可用于特征可视化、聚类或相似度计算。
        
        参数:
            x: torch.Tensor, 输入图像 [B*num_views, C, H, W]
            mode: str, 特征提取模式，可选：
                'base'  —— 仅提取 backbone 特征
                'task1' —— 任务1 分支特征 (结构分类)
                'task2' —— 任务2 分支特征 (空间群分类, 默认)
            with_normalize: bool, 是否进行 L2 归一化
        返回:
            features: torch.Tensor, 形状 [B, feature_dim]
        """
        # --- Step 1. 计算批量与视图数 ---
        batch_size = x.size(0) // self.num_views
        base_feat = self.features(x)  # [B*num_views, 512, 7, 7]

        # --- Step 2. 视图重组 ---
        _, C, H, W = base_feat.shape
        base_feat = base_feat.view(batch_size, self.num_views, C, H, W)  # [B, V, 512, 7, 7]

        # --- Step 3. 按 mode 选择特征路径 ---
        if mode == 'base':
            # 直接池化 backbone 输出
            pooled, _ = torch.max(base_feat, dim=1)  # [B, 512, 7, 7]
            features = F.adaptive_avg_pool2d(pooled, (1, 1)).flatten(1)

        elif mode == 'task1':
            # 与 forward 中一致的 task1 逻辑
            pooled, _ = torch.max(base_feat, dim=1)  # 多视角融合
            f = self.task1_head[0](pooled)  # AdaptiveAvgPool2d
            features = self.task1_head[1](f)  # Flatten [B, 512]

        elif mode == 'task2':
            # task2 路径需经过额外的特征变换层
            task2_base = self.task2_feature(base_feat.view(-1, C, H, W))
            _, C2, H2, W2 = task2_base.shape
            task2_feat = task2_base.view(batch_size, self.num_views, C2, H2, W2)
            pooled, _ = torch.max(task2_feat, dim=1)
            f = self.task2_head[0](pooled)  # AdaptiveAvgPool2d
            features = self.task2_head[1](f)  # Flatten [B, 256]

        else:
            raise ValueError("Invalid mode, choose from ['base', 'task1', 'task2']")

        # --- Step 4. L2 归一化 ---
        if with_normalize:
            features = F.normalize(features, p=2, dim=1)

        return features

    def forward(self, x):
        batch_size = x.size(0) // self.num_views
        base_feat = self.features(x)  # [batch*views, 512, 7, 7]

        # 任务1多视角融合（修改部分）
        _, C, H, W = base_feat.size()
        task1_feat = base_feat.view(batch_size, self.num_views, C, H, W)
        task1_pooled, _ = torch.max(task1_feat, dim=1)  # 沿视角维度取最大值
        y1 = self.task1_head(task1_pooled)

        # 任务2多视角融合（保持原有逻辑）
        task2_base = self.task2_feature(base_feat)
        _, C, H, W = task2_base.shape
        task2_feat = task2_base.view(batch_size, self.num_views, C, H, W)
        task2_pooled, _ = torch.max(task2_feat, dim=1)
        y2 = self.task2_head(task2_pooled)

        return y1, y2
    




# 辅助模块（保持不变）
class ViewAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, views, C, H, W = x.shape
        x_flat = x.reshape(batch*views, C, H, W)
        att = self.attn(x_flat).view(batch, views, 1, 1, 1)
        return x * att

class HybridPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = Parameter(torch.tensor(0.5))

    def forward(self, x):
        max_pool = torch.max(x, dim=1)[0]
        avg_pool = torch.mean(x, dim=1)
        return self.alpha*max_pool + (1-self.alpha)*avg_pool

# 验证测试
if __name__ == "__main__":
    # 单视角测试
    svBCNN = SVBCNN(name='resnet18')
    x = torch.randn(4, 3, 224, 224) 
    y1, y2 = svBCNN(x)
    print(f"SVBCNN输出: task1={y1.shape}, task2={y2.shape}") 
    
    # 多视角测试
    mvBCNN = MVBCNN(name='resnet188', svBCNN=svBCNN, num_views=2)
    x_mv = torch.randn(8, 3, 224, 224)  # 4个样本，每个2视角
    y1_mv, y2_mv = mvBCNN(x_mv)
    print(f"MVBCNN输出: task1={y1_mv.shape}, task2={y2_mv.shape}") 