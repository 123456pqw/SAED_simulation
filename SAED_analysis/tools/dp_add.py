import os
import io
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tools.run_saed import run_ed_simulation   # 你已有的接口
from models.mvbcnn import SVCNN, MVCNN
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

# ============= 辅助函数 ================= #
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

def _cleanup_temp_files(temp_files):
    """清理临时文件"""
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        except Exception as e:
            print(f"⚠️ 清理临时文件失败 {temp_file}: {e}")

# ============= 模拟 + 写入数据库 ================= #
def simulate_and_add_to_database(cif_input, sg, mid, data_root="/share/SAED_analysis/data", 
                                delete_temp: bool = True):
    """
    输入一个 CIF 文件：
    1. 自动模拟两个固定视角 beam_0_0_1 & beam_1_0_0
    2. 按你数据库格式写入 data/sg/mid
    
    参数:
        cif_input: CIF文件路径、UploadFile对象或二进制数据
        sg: 空间群编号
        mid: 材料ID
        data_root: 数据保存根目录
        delete_temp: 是否删除临时文件
    """
    temp_files = []
    
    try:
        # 处理CIF输入
        cif_path = _process_cif_input(cif_input)
        if isinstance(cif_input, (UploadFile, bytes)):
            temp_files.append(cif_path)

        # 1. 创建目标目录
        target_dir = os.path.join(data_root, str(sg), str(mid))
        os.makedirs(target_dir, exist_ok=True)

        # 2. 定义两个固定视角
        beam_list = {
            "beam_0_0_1.png": [0, 0, 1],
            "beam_1_0_0.png": [1, 0, 0]
        }

        # 3. 逐个模拟
        for fname, beam in beam_list.items():
            save_path = os.path.join(target_dir, fname)
            run_ed_simulation(
                cif_path,
                zone_axis=beam,
                filename=save_path
            )

        print(f"✅ SAED patterns saved to: {target_dir}")
        return target_dir
        
    finally:
        # 清理临时文件
        if delete_temp:
            _cleanup_temp_files(temp_files)

# ============= 特征提取 ================= #
def load_image_from_path_or_array(image_path_or_array, transform):
    """支持文件路径或numpy数组作为输入"""
    if isinstance(image_path_or_array, np.ndarray):
        # 直接处理numpy数组
        im = Image.fromarray(image_path_or_array)
    elif isinstance(image_path_or_array, bytes):
        # 处理二进制图像数据
        im = Image.open(io.BytesIO(image_path_or_array)).convert('RGB')
    elif isinstance(image_path_or_array, UploadFile):
        # 处理UploadFile图像
        contents = image_path_or_array.file.read()
        im = Image.open(io.BytesIO(contents)).convert('RGB')
    else:
        # 文件路径
        im = Image.open(image_path_or_array).convert('RGB')
    
    return transform(im)

@torch.no_grad()
def extract_feature_from_dir_or_images(mat_input, mvcnn, device, is_dir: bool = True):
    """
    从目录或直接从图像数据提取特征
    
    参数:
        mat_input: 目录路径或包含两个图像的列表/元组
        mvcnn: MVCNN模型
        device: 计算设备
        is_dir: 是否为目录输入
    """
    transform = transforms.Compose([
        transforms.CenterCrop(300),
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    images = []
    
    if is_dir:
        # 从目录加载图像
        view_files = ["beam_0_0_1.png", "beam_1_0_0.png"]
        for vf in view_files:
            img_path = os.path.join(mat_input, vf)
            if not os.path.exists(img_path):
                print(f"❌ Missing {img_path}, skip.")
                return None
            images.append(load_image_from_path_or_array(img_path, transform))
    else:
        # 直接使用提供的图像数据
        if len(mat_input) != 2:
            print(f"❌ 需要提供两个视角的图像，当前提供了 {len(mat_input)} 个")
            return None
        
        for img_data in mat_input:
            images.append(load_image_from_path_or_array(img_data, transform))

    x = torch.stack(images, dim=0).to(device)
    feature = mvcnn.extract_feature(x)
    
    if isinstance(feature, tuple):
        feature = feature[0]
        
    return feature.cpu()

# ============= 封装成一个“一键扩展数据库”函数 ================= #
async def add_cif_to_database(cif_input, sg, mid,
                        data_root="/share/SAED_analysis/data",
                        feat_root="/share/SAED_analysis/features_tripletv3",
                        delete_temp: bool = True):
    """
    输入 CIF → 模拟两个视角 → 提取特征 → 写入 features 数据库
    
    参数:
        cif_input: CIF文件路径、UploadFile对象或二进制数据
        sg: 空间群编号
        mid: 材料ID
        data_root: 数据保存根目录
        feat_root: 特征保存根目录
        delete_temp: 是否删除临时文件
    """
    temp_files = []
    
    try:
        # 1. 先模拟
        mat_dir = simulate_and_add_to_database(cif_input, sg, mid, data_root=data_root, 
                                             delete_temp=delete_temp)

        # 2. 再提取特征
        feat = extract_feature_from_dir_or_images(mat_dir, mvcnn, device, is_dir=True)
        if feat is None:
            print(f"❌ Failed to extract feature for {mid}.")
            return None

        # 3. 保存特征
        save_dir = os.path.join(feat_root, str(sg))
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{mid}.npy")
        torch.save(feat, save_path)

        print(f"✅ Feature saved to: {save_path}")
        return save_path
        
    finally:
        # 清理临时文件
        if delete_temp:
            _cleanup_temp_files(temp_files)

# ============= 扩展功能：直接从图像数据添加 ================= #
def add_images_to_database(images_input, sg, mid,
                          data_root="/share/SAED_analysis/data",
                          feat_root="/share/SAED_analysis/features",
                          save_images: bool = True):
    """
    直接从图像数据添加到数据库（不需要CIF文件）
    
    参数:
        images_input: 包含两个图像的列表/元组（beam_0_0_1, beam_1_0_0）
                     支持路径、numpy数组、二进制数据或UploadFile
        sg: 空间群编号
        mid: 材料ID
        data_root: 数据保存根目录
        feat_root: 特征保存根目录
        save_images: 是否保存图像到数据库
    """
    # 如果需要保存图像
    if save_images:
        target_dir = os.path.join(data_root, str(sg), str(mid))
        os.makedirs(target_dir, exist_ok=True)
        
        view_files = ["beam_0_0_1.png", "beam_1_0_0.png"]
        
        for img_data, fname in zip(images_input, view_files):
            save_path = os.path.join(target_dir, fname)
            
            if isinstance(img_data, np.ndarray):
                # numpy数组
                img = Image.fromarray(img_data)
                img.save(save_path)
            elif isinstance(img_data, bytes):
                # 二进制数据
                img = Image.open(io.BytesIO(img_data))
                img.save(save_path)
            elif isinstance(img_data, UploadFile):
                # UploadFile
                contents = img_data.file.read()
                with open(save_path, 'wb') as f:
                    f.write(contents)
            else:
                # 文件路径，复制文件
                import shutil
                shutil.copy2(img_data, save_path)
    
    # 提取特征
    feat = extract_feature_from_dir_or_images(images_input, mvcnn, device, is_dir=False)
    if feat is None:
        print(f"❌ Failed to extract feature for {mid}.")
        return None

    # 保存特征
    save_dir = os.path.join(feat_root, str(sg))
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{mid}.npy")
    torch.save(feat, save_path)

    print(f"✅ Feature saved to: {save_path}")
    return save_path

# ============= 使用示例 ================= #
if __name__ == "__main__":
    # 示例1：使用文件路径
    # add_cif_to_database(
    #     cif_input="/share/SAED_analysis/utils/betaFe2O3.cif",
    #     sg=167,
    #     mid="mp-12345"
    # )
    
    # 示例2：直接使用图像数据
    # img1 = cv2.imread("beam_0_0_1.png")
    # img2 = cv2.imread("beam_1_0_0.png")
    # add_images_to_database(
    #     images_input=[img1, img2],
    #     sg=167,
    #     mid="mp-67890"
    # )
    pass