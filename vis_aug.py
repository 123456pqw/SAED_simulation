import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

# ------------------------- 数据增强函数 -------------------------
def add_noise(image, poisson_scale=0.05, gaussian_sigma=0.02):
    """
    添加混合噪声：泊松 + 高斯
    :param image: 输入图像，np.array，范围 [0,1]
    :param poisson_scale: 泊松噪声比例
    :param gaussian_sigma: 高斯噪声标准差
    :return: 加噪后的图像，np.array，范围 [0,1]
    """
    # 确保浮点数
    image = image.astype(np.float32)
    # 泊松噪声
    image_255 = np.clip(image*255, 0, 255)
    noisy_poisson = np.random.poisson(image_255 * poisson_scale) / 255.0
    # 高斯噪声
    gaussian_noise = np.random.normal(0, gaussian_sigma, image.shape)
    noisy = np.clip(image + noisy_poisson + gaussian_noise, 0, 1)
    return noisy

def random_crop(image, crop_ratio=0.8):
    """随机裁剪增强"""
    w, h = image.size
    new_w = int(w * crop_ratio)
    new_h = int(h * crop_ratio)
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    return image.crop((left, top, left+new_w, top+new_h))

def random_rotate(image, max_angle=45):
    """带黑色背景的随机旋转增强"""
    angle = random.uniform(-max_angle, max_angle)
    return image.rotate(angle, expand=False, fillcolor=(0,0,0), resample=Image.BICUBIC)

# ------------------------- 可视化与保存 -------------------------
def visualize_augmentations(image_path, noise_params, num_samples=3, output_dir="augmented_images"):
    """
    可视化并保存不同增强结果
    :param image_path: 原始图像路径
    :param noise_params: 噪声参数列表 [{'poisson':..., 'gaussian':...}, ...]
    :param num_samples: 每种增强展示的样本数量
    :param output_dir: 保存目录
    """
    os.makedirs(output_dir, exist_ok=True)
    orig_image = Image.open(image_path)
    #orig_image = orig_image.crop((30, 30, 430, 430)) 
    w, h = orig_image.size

    fig = plt.figure(figsize=(18, 12))
    
    # ----------------- 原始图像 -----------------
    ax = fig.add_subplot(4, num_samples+1, 1)
    ax.imshow(orig_image)
    ax.set_title("Original")
    ax.axis('off')

    # ----------------- 噪声增强 -----------------
    for i, params in enumerate(noise_params[:num_samples], start=2):
        ax = fig.add_subplot(4, num_samples+1, i)
        image_np = np.array(orig_image).astype(np.float32)/255.0
        noisy_img = add_noise(image_np, poisson_scale=params['poisson'], gaussian_sigma=params['gaussian'])
        noisy_uint8 = (noisy_img*255).astype(np.uint8)
        noisy_pil = Image.fromarray(noisy_uint8)
        noisy_pil.save(os.path.join(output_dir, f"noise_{i-1}_P{params['poisson']}_G{params['gaussian']}.png"))
        ax.imshow(noisy_pil)
        ax.set_title(f"Noise {i-1}\nP={params['poisson']}, G={params['gaussian']}")
        ax.axis('off')

    # ----------------- 随机裁剪 -----------------
    for i in range(num_samples+1):
        ax = fig.add_subplot(4, num_samples+1, (num_samples+1)*1 + i +1)
        if i==0:
            ax.imshow(orig_image)
            ax.set_title("Original")
        else:
            crop_ratio = random.uniform(0.6, 0.9)
            crop_img = random_crop(orig_image, crop_ratio)
            crop_img.save(os.path.join(output_dir, f"crop_{i}_ratio{crop_ratio:.2f}.png"))
            ax.imshow(crop_img)
            ax.set_title(f"Crop {i}\nRatio={crop_ratio:.2f}")
        ax.axis('off')

    # ----------------- 随机旋转 -----------------
    for i in range(num_samples+1):
        ax = fig.add_subplot(4, num_samples+1, (num_samples+1)*2 + i +1)
        if i==0:
            ax.imshow(orig_image)
            ax.set_title("Original")
        else:
            rot_img = random_rotate(orig_image)
            rot_img.save(os.path.join(output_dir, f"rotate_{i}.png"))
            ax.imshow(rot_img)
            ax.set_title(f"Rotate {i}")
        ax.axis('off')

    # ----------------- 单独噪声类型 -----------------
    for i in range(num_samples+1):
        ax = fig.add_subplot(4, num_samples+1, (num_samples+1)*3 + i +1)
        image_np = np.array(orig_image).astype(np.float32)/255.0
        if i==0:
            ax.imshow(orig_image)
            ax.set_title("Original")
        else:
            if i==1:
                noisy = add_noise(image_np, poisson_scale=0.1, gaussian_sigma=0)
                fname = "poisson_only.png"
                title = "Poisson Only"
            elif i==2:
                noisy = add_noise(image_np, poisson_scale=0, gaussian_sigma=0.05)
                fname = "gaussian_only.png"
                title = "Gaussian Only"
            else:
                noisy = add_noise(image_np, poisson_scale=0.05, gaussian_sigma=0.02)
                fname = "mixed_noise.png"
                title = "Mixed Noise"
            Image.fromarray((noisy*255).astype(np.uint8)).save(os.path.join(output_dir, fname))
            ax.imshow((noisy*255).astype(np.uint8))
            ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "augmentations_overview.png"), dpi=300)
    plt.show()

# ------------------------- 主函数 -------------------------
if __name__ == "__main__":
    noise_configs = [
        {'poisson': 0.05, 'gaussian': 0.01},
        {'poisson': 0.02, 'gaussian': 0.03},
        {'poisson': 0.01, 'gaussian': 0.05}
    ]

    visualize_augmentations(
        image_path='/internfs/pengqianwen/MVBCNN/data/125/mp-628185/beam_0_0_1.png',
        noise_params=noise_configs,
        num_samples=5,
        output_dir="augmented_001"
    )

    visualize_augmentations(
        image_path='/internfs/pengqianwen/MVBCNN/data/125/mp-628185/beam_2_0_1.png',
        noise_params=noise_configs,
        num_samples=5,
        output_dir="augmented_201"
    )
