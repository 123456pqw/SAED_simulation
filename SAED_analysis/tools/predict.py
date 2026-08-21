import os
import io
import torch
# import asyncio
import pandas as pd
import sys
sys.path.append('/share')
from SAED_analysis.models.mvbcnn import MVCNN, SVCNN
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Union, List, Dict, Any, Optional
import tempfile
from starlette.datastructures import UploadFile
from SAED_analysis.utils.picprocess import *
import torch.multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True)

# ---------- 图像处理函数 ----------
def load_image_from_data_async(image_data, transform=None):
    """
    异步从多种数据源加载图像
    """
    if isinstance(image_data, np.ndarray):
        img = Image.fromarray(image_data).convert('RGB')
    elif isinstance(image_data, bytes):
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
    elif isinstance(image_data, UploadFile):
        # 异步读取UploadFile
        contents = image_data.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
    elif hasattr(image_data, 'file'):
        # Gradio文件对象
        img = Image.open(image_data.name).convert('RGB')
    elif isinstance(image_data, tempfile._TemporaryFileWrapper):
        img = Image.open(image_data.name).convert('RGB')
    else:
        # 文件路径
        if not os.path.exists(image_data):
            raise FileNotFoundError(f"图像路径不存在：{image_data}")
        img = Image.open(image_data).convert('RGB')
    
    # 图像处理
    img = img.crop((50, 50, 350, 350)).resize((1024, 1024), Image.BICUBIC)
    if transform:
        img = transform(img)
    
    return img

def load_image_async(image_inputs, transform=None):
    """
    异步加载单视角/多视角图像
    """
    if not isinstance(image_inputs, list):
        image_inputs = [image_inputs]
    
    images = []
    input_names = []
    
    for img_data in image_inputs:
        if img_data is None:
            continue
            
        img = load_image_from_data_async(img_data, transform)
        images.append(img)
        
        # 记录输入名称
        if isinstance(img_data, str) and os.path.exists(img_data):
            input_names.append(os.path.basename(img_data))
        elif hasattr(img_data, 'filename'):
            input_names.append(img_data.filename)
        elif hasattr(img_data, 'name'):
            input_names.append(os.path.basename(getattr(img_data, 'name', 'image')))
        else:
            input_names.append(f"image_{len(input_names)+1}")
    
    if not images:
        raise ValueError("没有提供有效的图像数据")
    
    return torch.stack(images), input_names

# -------------------------- 推理函数 --------------------------
def infer_spacegroup_async(
    model, 
    image_inputs, 
    real_sg_list,
    device="cuda", 
    top_k=5, 
    is_multi_view=False
):
    """
    异步推理函数
    """
    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 异步加载图像
    img_tensor, input_names = load_image_async(image_inputs, transform=transform)
    img_tensor = img_tensor.to(device, non_blocking=True)
    
    # 模型推理（在线程池中运行同步代码）
    def run_inference():
        model.eval()
        with torch.no_grad():
            if is_multi_view:
                V, C, H, W = img_tensor.size()
                img_tensor_reshaped = img_tensor.view(-1, C, H, W).cuda()
                output = model(img_tensor_reshaped)
                if isinstance(output, tuple):
                    logits_h2 = output[1]
                else:
                    logits_h2 = output
                probs = torch.softmax(logits_h2, dim=1)
            else:
                output = model(img_tensor)
                if isinstance(output, tuple):
                    logits_h2 = output[1]
                else:
                    logits_h2 = output
                probs = torch.softmax(logits_h2, dim=1)
        
        return probs
    
    # 在异步环境中运行同步推理
    # probs = asyncio.get_event_loop().run_in_executor(None, run_inference)
    probs = run_inference()
    # 处理结果
    probs = probs.cpu().numpy().squeeze()
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_sg = [real_sg_list[idx] for idx in top_indices]
    #top_sg[0]=14
    top_probs = [round(float(probs[idx]), 4) for idx in top_indices]
    
    result = {
        "前5个预测空间群": top_sg,
        "对应概率": top_probs,
        "最可能空间群": top_sg[0],
        "最高概率": top_probs[0]
    }
    
    return result

# =====================================================
# 全局缓存
# =====================================================
_GLOBAL_SVCNN = None
_GLOBAL_MVCNN = None
_GLOBAL_CLASSNAMES = None

def _load_classnames(csv_path="/share/SAED_analysis/utils/data.csv"):
    """加载空间群编号列表"""
    global _GLOBAL_CLASSNAMES
    if _GLOBAL_CLASSNAMES is None:
        df = pd.read_csv(csv_path)
        _GLOBAL_CLASSNAMES = df.iloc[:, 0].tolist()
    return _GLOBAL_CLASSNAMES

def _load_svcnn(ckpt):
    """加载 SVCNN"""
    global _GLOBAL_SVCNN
    if _GLOBAL_SVCNN is None:
        model = SVCNN("GVCNN", pretraining=False, cnn_name="inception")
        model.load_state_dict(torch.load(ckpt, map_location="cuda"))
        model = model.cuda().eval()
        _GLOBAL_SVCNN = model
    return _GLOBAL_SVCNN

def _load_mvcnn(svcnn_ckpt, mvcnn_ckpt):
    """加载 MVCNN"""
    global _GLOBAL_MVCNN
    if _GLOBAL_MVCNN is None:
        sv = SVCNN("GVCNN", pretraining=False, cnn_name="inception")
        sv.load_state_dict(torch.load(svcnn_ckpt, map_location="cuda"))
        sv = sv.cuda()

        mv = MVCNN("GVCNN", sv)
        mv.load_state_dict(torch.load(mvcnn_ckpt, map_location="cuda"))
        mv = mv.cuda().eval()

        _GLOBAL_MVCNN = mv
    return _GLOBAL_MVCNN

# =====================================================
# 主服务函数（专门支持两张图片上传）
# =====================================================

def pre_symmetry(
    image1: Optional[Union[str, np.ndarray, UploadFile]] = None,
    image2: Optional[Union[str, np.ndarray, UploadFile]] = None,
    model_type: str = "mvbcnn",
    top_k: int = 5,
    svcnn_ckpt: str = "/share/SAED_analysis/models_pt/svbcnn/model-00083.pth",
    mvcnn_ckpt: str = "/share/SAED_analysis/models_pt/mvbcnn/model-00023.pth",
    device: str = "cuda"
):
    """
    🧠 pre_symmetry() → 空间群预测服务（支持上传两张图片）
    ------------------------------------------------------
    输入：
        - image1: 第一张图片（必填）
        - image2: 第二张图片（可选）
        - model_type: "svcnn" (单图) / "mvcnn" (多图)
        - top_k: 返回前 k 个空间群
    输出（dict）：
        包含预测结果的字典
    """
    # 验证输入
    if image1 is None:
        raise ValueError("必须提供至少一张图片（image1）")
    
    # 设置设备
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("⚠️ CUDA不可用，使用CPU运行")
    
    # 准备图像输入列表
    image_inputs = [image1]
    if image2 is not None:
        image_inputs.append(image2)
    
    # 选择模型
    if model_type == "svcnn" or image2 is None:
        model = _load_svcnn(svcnn_ckpt)
        multi_view = False
        if image2 is not None:
            print("⚠️ SVCNN只支持单图输入，将只使用第一张图片")
            image_inputs = [image1]
    else:
        model = _load_mvcnn(svcnn_ckpt, mvcnn_ckpt)
        multi_view = True
    
    # 加载空间群列表
    real_sg_list = _load_classnames()
    
    # 执行推理
    result = infer_spacegroup_async(
        model=model,
        image_inputs=image_inputs,
        real_sg_list=real_sg_list,
        device=device,
        top_k=top_k,
        is_multi_view=multi_view
    )
    
    # 格式化输出
    clean_output = {
        "top_k_sg": result["前5个预测空间群"],
        "top_k_prob": result["对应概率"],
        "best_sg": result["最可能空间群"],
        #"best_prob": result["最高概率"],
        "model_used": model_type
    }
    
    return clean_output
