import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from models.MVBCNN_18_a_new import SVBCNN,MVBCNN

def load_pretrained_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # 去掉多余前缀 "net."
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("net."):
            new_k = k[len("net."):]
        else:
            new_k = k
        new_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ Loaded weights with {len(missing)} missing and {len(unexpected)} unexpected keys.")
    if missing:
        print("Missing:", missing)
    if unexpected:
        print("Unexpected:", unexpected)

# ========== 用户自定义部分 ==========
input_root = ""        # 原始 SAED 图像路径
output_root = ""   # 编码特征保存路径
num_views = 2                                            # 视角数量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 已加载好的 MVBCNN 模型
model = SVBCNN("GVBCNN", pretraining=False, BCNN_name="inception")
load_pretrained_weights(model, "")
mvBCNN = MVBCNN("GVBCNN", model)
mvBCNN.load_state_dict(torch.load(""))
mvBCNN.eval()
mvBCNN.to(device)

# 图像预处理
transform = transforms.Compose([
    transforms.CenterCrop(300),
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ========== 特征提取函数 ==========
@torch.no_grad()
def extract_feature_for_material(material_dir):
    """输入一个材料目录，输出拼接后的 descriptor tensor"""
    view_files = ["beam_0_0_1.png", "beam_1_0_0.png"]
    images = []

    for vf in view_files:
        img_path = os.path.join(material_dir, vf)
        if not os.path.exists(img_path):
            return None  # 缺少视图则跳过

        im = Image.open(img_path).convert('RGB')
        im = transform(im)
        images.append(im)

    x = torch.stack(images, dim=0).to(device)  # [num_views, 3, H, W]
    feature = mvBCNN.extract_feature(x)         # 假设输出为 [512] 或 [1, 512]
    if isinstance(feature, tuple):
        feature = feature[0]
    return feature.cpu()


# ========== 主循环 ==========
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

        # 构建输出目录
        save_dir = os.path.join(output_root, sg)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{mid}.npy")

        # 保存特征向量
        torch.save(feat, save_path)
