"""
ControlNet Inpainting 推理脚本

使用训练好的 ControlNet 模型进行图像补全。

推理流程：
1. 加载训练好的 ControlNet 模型
2. 加载待补全的图像和 mask
3. 将图像编码到 latent space
4. 从随机噪声开始，逐步去噪
5. 将 latent 解码回图像空间
6. 保存补全结果

支持：
- DDPM 采样器（50 步）
- DDIM 采样器（20 步，更快）
- 批量推理
- 可视化对比图
"""

import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from tqdm import tqdm

from model import create_model


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ControlNet Inpainting 推理")
    
    # 模型参数
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="ControlNet checkpoint 路径（.pth 文件）"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="预训练 Stable Diffusion 模型路径"
    )
    
    # 输入输出
    parser.add_argument(
        "--input_image",
        type=str,
        required=True,
        help="待补全的图像路径"
    )
    parser.add_argument(
        "--input_mask",
        type=str,
        required=True,
        help="mask 图像路径（白色区域将被补全）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/inference",
        help="输出目录"
    )
    
    # 推理参数
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="推理步数（DDPM=50, DDIM=20）"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="引导强度（通常设为 1.0，不使用 classifier-free guidance）"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ddpm",
        choices=["ddpm", "ddim"],
        help="采样器类型"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备"
    )
    
    args = parser.parse_args()
    return args


def load_image(image_path, size=512):
    """
    加载并预处理图像
    
    参数：
    - image_path: str, 图像路径
    - size: int, 目标尺寸
    
    返回：
    - image_tensor: [1, 3, H, W], 归一化到 [-1, 1]
    - original_pil: PIL.Image, 原始图像（用于可视化）
    """
    image = Image.open(image_path).convert('RGB')
    original_pil = image.copy()
    
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]
    return image_tensor, original_pil


def load_mask(mask_path, size=512):
    """
    加载并预处理 mask
    
    参数：
    - mask_path: str, mask 路径
    - size: int, 目标尺寸
    
    返回：
    - mask_tensor: [1, 1, H, W], 二值 mask（0 或 1）
    - mask_pil: PIL.Image, 原始 mask（用于可视化）
    """
    mask = Image.open(mask_path).convert('L')
    mask_pil = mask.copy()
    
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    
    mask_tensor = transform(mask).unsqueeze(0)  # [1, 1, H, W]
    mask_tensor = (mask_tensor > 0.5).float()  # 二值化
    
    return mask_tensor, mask_pil


def denormalize(tensor):
    """
    将 [-1, 1] 的 tensor 转换为 [0, 1]
    
    参数：
    - tensor: torch.Tensor
    
    返回：
    - tensor: torch.Tensor, [0, 1] 范围
    """
    return (tensor + 1.0) / 2.0


def tensor_to_pil(tensor):
    """
    将 tensor 转换为 PIL Image
    
    参数：
    - tensor: [C, H, W], 范围 [0, 1]
    
    返回：
    - image: PIL.Image
    """
    tensor = tensor.cpu().clamp(0, 1)
    array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


@torch.no_grad()
def infer(args):
    """
    主推理函数
    
    参数：
    - args: 命令行参数
    """
    print("=" * 80)
    print("ControlNet Inpainting 推理")
    print("=" * 80)
    print(f"设备: {args.device}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"输入图像: {args.input_image}")
    print(f"输入 mask: {args.input_mask}")
    print(f"采样器: {args.scheduler}")
    print(f"推理步数: {args.num_inference_steps}")
    print("=" * 80)
    
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    print("\n正在加载模型...")
    model = create_model(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        freeze_unet=True
    )
    
    # 加载 ControlNet 权重
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    model.controlnet.load_state_dict(checkpoint)
    print(f"✓ ControlNet 权重已加载")
    
    model = model.to(args.device)
    model.eval()
    
    # 加载 VAE
    print("正在加载 VAE...")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae"
    )
    vae = vae.to(args.device)
    vae.eval()
    
    # 加载调度器
    if args.scheduler == "ddpm":
        scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="scheduler"
        )
    else:  # ddim
        scheduler = DDIMScheduler.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="scheduler"
        )
    
    # 设置推理步数
    scheduler.set_timesteps(args.num_inference_steps)
    
    # 加载输入图像和 mask
    print("\n正在加载输入...")
    image_tensor, image_pil = load_image(args.input_image, size=512)
    mask_tensor, mask_pil = load_mask(args.input_mask, size=512)
    
    image_tensor = image_tensor.to(args.device)
    mask_tensor = mask_tensor.to(args.device)
    
    # 生成 masked_image
    masked_image_tensor = image_tensor * (1 - mask_tensor) + mask_tensor  # mask 区域为白色
    
    print(f"图像形状: {image_tensor.shape}")
    print(f"Mask 形状: {mask_tensor.shape}")
    
    # 编码到 latent space
    print("\n正在编码图像...")
    with torch.no_grad():
        # 编码 masked_image
        masked_latent = vae.encode(masked_image_tensor).latent_dist.sample()
        masked_latent = masked_latent * vae.config.scaling_factor
        
        # mask 下采样到 latent 尺寸
        mask_latent = F.interpolate(
            mask_tensor,
            size=(masked_latent.shape[2], masked_latent.shape[3]),
            mode='nearest'
        )
    
    print(f"Latent 形状: {masked_latent.shape}")
    
    # 构建 ControlNet 输入
    mask_latent_4ch = mask_latent.repeat(1, 4, 1, 1)
    control_input = masked_latent + mask_latent_4ch * 0.1
    
    # 初始化随机噪声
    print("\n开始去噪过程...")
    latents = torch.randn_like(masked_latent)
    
    # 去噪循环
    for i, t in enumerate(tqdm(scheduler.timesteps, desc="去噪")):
        # 扩展时间步
        timestep = t.unsqueeze(0).to(args.device)
        
        # 预测噪声
        noise_pred = model(
            x_t=latents,
            timestep=timestep,
            control_input=control_input,
            encoder_hidden_states=None
        )
        
        # 去噪步骤
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # 解码 latents 到图像空间
    print("\n正在解码图像...")
    with torch.no_grad():
        latents = latents / vae.config.scaling_factor
        output_image = vae.decode(latents).sample
    
    # 后处理
    output_image = denormalize(output_image)
    output_pil = tensor_to_pil(output_image[0])
    
    # 保存结果
    output_path = os.path.join(args.output_dir, "inpainted.png")
    output_pil.save(output_path)
    print(f"\n✓ 补全结果已保存到: {output_path}")
    
    # 创建对比图
    print("正在生成对比图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 原图（masked）
    masked_image_vis = denormalize(masked_image_tensor[0])
    axes[0, 0].imshow(tensor_to_pil(masked_image_vis))
    axes[0, 0].set_title('Input (Masked Image)', fontsize=14)
    axes[0, 0].axis('off')
    
    # Mask
    axes[0, 1].imshow(mask_pil, cmap='gray')
    axes[0, 1].set_title('Mask (white = inpaint)', fontsize=14)
    axes[0, 1].axis('off')
    
    # 补全结果
    axes[1, 0].imshow(output_pil)
    axes[1, 0].set_title('Inpainted Result', fontsize=14)
    axes[1, 0].axis('off')
    
    # 如果有原始完整图像，显示对比
    # 这里我们显示补全结果的放大版
    axes[1, 1].imshow(output_pil)
    axes[1, 1].set_title('Result (Zoom)', fontsize=14)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(args.output_dir, "comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"✓ 对比图已保存到: {comparison_path}")
    
    print("\n" + "=" * 80)
    print("推理完成！")
    print("=" * 80)


if __name__ == "__main__":
    args = parse_args()
    infer(args)

