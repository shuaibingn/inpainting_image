"""
ControlNet Inpainting 训练脚本

基于 lllyasviel/ControlNet 的训练流程，实现图像补全模型的训练。

训练流程：
1. 加载 COCO 数据集
2. 初始化 ControlNet 模型
3. 对每个 batch：
   - 用 VAE 将图像编码到 latent space
   - 采样时间步 t
   - 对 latent 加噪
   - 前向传播预测噪声
   - 计算 MSE 损失
   - 反向传播更新 ControlNet 参数
4. 定期保存 checkpoint

训练配置：
- 优化器：AdamW (lr=1e-5)
- 学习率调度：CosineAnnealing
- 支持混合精度训练
- 仅训练 ControlNet，冻结 UNet
"""

import os
import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from diffusers import AutoencoderKL, DDPMScheduler
from accelerate import Accelerator
from accelerate.utils import set_seed

from data import create_dataloader
from model import create_model


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练 ControlNet Inpainting 模型")
    
    # 数据集参数
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="COCO 图像文件夹路径（如 train2017）"
    )
    parser.add_argument(
        "--ann_file",
        type=str,
        required=True,
        help="COCO 标注文件路径（如 instances_train2017.json）"
    )
    
    # 模型参数
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="预训练 Stable Diffusion 模型路径"
    )
    
    # 训练参数
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="训练批次大小"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="训练轮数"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="学习率"
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="AdamW 权重衰减"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="数据加载线程数"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    # 保存和日志参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="输出目录（保存 checkpoint 和日志）"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="每隔多少步保存一次 checkpoint"
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=50,
        help="每隔多少步记录一次日志"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="混合精度训练（no/fp16/bf16）"
    )
    
    # 数据增强参数
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="图像尺寸"
    )
    parser.add_argument(
        "--use_irregular_mask",
        action="store_true",
        default=True,
        help="是否使用不规则 mask"
    )
    
    args = parser.parse_args()
    return args


def train(args):
    """
    主训练函数
    
    参数：
    - args: 命令行参数
    """
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"controlnet_inpainting_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化 Accelerator（用于分布式训练和混合精度）
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=1,
        log_with="tensorboard",
        project_dir=output_dir,
    )
    
    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)
    
    # 在主进程中打印信息
    if accelerator.is_main_process:
        print("=" * 80)
        print("ControlNet Inpainting 训练")
        print("=" * 80)
        print(f"输出目录: {output_dir}")
        print(f"设备: {accelerator.device}")
        print(f"混合精度: {args.mixed_precision}")
        print(f"批次大小: {args.batch_size}")
        print(f"学习率: {args.learning_rate}")
        print(f"训练轮数: {args.num_epochs}")
        print("=" * 80)
    
    # 创建数据加载器
    if accelerator.is_main_process:
        print("\n正在加载数据集...")
    
    dataloader = create_dataloader(
        image_dir=args.image_dir,
        ann_file=args.ann_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        image_size=args.image_size,
        use_irregular_mask=args.use_irregular_mask,
    )
    
    if accelerator.is_main_process:
        print(f"数据集大小: {len(dataloader.dataset)}")
        print(f"批次数量: {len(dataloader)}")
    
    # 创建模型
    if accelerator.is_main_process:
        print("\n正在加载模型...")
    
    model = create_model(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        freeze_unet=True
    )
    
    # 加载 VAE（用于编码图像到 latent space）
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae"
    )
    vae.requires_grad_(False)  # VAE 不需要训练
    vae.eval()
    
    # 加载噪声调度器
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )
    
    # 创建优化器
    # 只优化 ControlNet 的参数
    optimizer = torch.optim.AdamW(
        model.controlnet.parameters(),
        lr=args.learning_rate,
        weight_decay=args.adam_weight_decay,
    )
    
    # 学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs * len(dataloader),
        eta_min=1e-7
    )
    
    # 使用 Accelerator 准备模型、优化器和数据加载器
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    
    # VAE 移到设备上（不需要用 accelerator.prepare）
    vae = vae.to(accelerator.device)
    
    # TensorBoard
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))
    
    # 训练循环
    if accelerator.is_main_process:
        print("\n开始训练...")
        print("=" * 80)
    
    global_step = 0
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # 进度条
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            disable=not accelerator.is_local_main_process
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            original_image, masked_image, mask = batch
            
            # 将图像移到设备
            original_image = original_image.to(accelerator.device)
            masked_image = masked_image.to(accelerator.device)
            mask = mask.to(accelerator.device)
            
            # 使用 VAE 编码图像到 latent space
            with torch.no_grad():
                # 编码原图
                latent_dist = vae.encode(original_image).latent_dist
                latents = latent_dist.sample() * vae.config.scaling_factor
                
                # 编码 masked_image
                masked_latent_dist = vae.encode(masked_image).latent_dist
                masked_latents = masked_latent_dist.sample() * vae.config.scaling_factor
                
                # mask 下采样到 latent 尺寸
                mask_latent = F.interpolate(
                    mask, 
                    size=(latents.shape[2], latents.shape[3]),
                    mode='nearest'
                )
            
            # 采样随机时间步
            timesteps = torch.randint(
                0, 
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device
            ).long()
            
            # 生成随机噪声
            noise = torch.randn_like(latents)
            
            # 对 latents 加噪（前向扩散过程）
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # 构建 ControlNet 输入：[masked_latent, mask_latent]
            # 注意：这里的 masked_latents 是 4 通道，mask_latent 是 1 通道
            # 但模型期望 4 通道输入，所以我们将 mask 扩展到 4 通道
            # 或者拼接后再处理
            # 这里我们简单地将 mask 复制 4 次
            mask_latent_4ch = mask_latent.repeat(1, 4, 1, 1)
            control_input = masked_latents + mask_latent_4ch * 0.1  # 轻微混合
            
            # 前向传播
            noise_pred = model(
                x_t=noisy_latents,
                timestep=timesteps,
                control_input=control_input,
                encoder_hidden_states=None  # 不使用文本条件
            )
            
            # 计算损失（预测噪声 vs 真实噪声）
            loss = F.mse_loss(noise_pred, noise, reduction="mean")
            
            # 反向传播
            accelerator.backward(loss)
            
            # 梯度裁剪
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            
            # 优化器步进
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # 累积损失
            epoch_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{lr_scheduler.get_last_lr()[0]:.2e}'
            })
            
            # 记录日志
            if global_step % args.log_steps == 0 and accelerator.is_main_process:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], global_step)
            
            # 保存 checkpoint
            if global_step % args.save_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    
                    # 保存 ControlNet
                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save(
                        unwrapped_model.controlnet.state_dict(),
                        os.path.join(save_path, "controlnet.pth")
                    )
                    
                    # 保存优化器状态
                    torch.save(
                        optimizer.state_dict(),
                        os.path.join(save_path, "optimizer.pth")
                    )
                    
                    # 保存训练状态
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'args': args,
                    }, os.path.join(save_path, "training_state.pth"))
                    
                    print(f"\n✓ Checkpoint 已保存到: {save_path}")
            
            global_step += 1
        
        # Epoch 结束统计
        avg_epoch_loss = epoch_loss / len(dataloader)
        
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch + 1}/{args.num_epochs} 完成")
            print(f"平均损失: {avg_epoch_loss:.4f}")
            print(f"学习率: {lr_scheduler.get_last_lr()[0]:.2e}")
            print("-" * 80)
            
            writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch)
    
    # 训练结束，保存最终模型
    if accelerator.is_main_process:
        print("\n训练完成！正在保存最终模型...")
        
        final_save_path = os.path.join(output_dir, "final_model")
        os.makedirs(final_save_path, exist_ok=True)
        
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(
            unwrapped_model.controlnet.state_dict(),
            os.path.join(final_save_path, "controlnet.pth")
        )
        
        print(f"✓ 最终模型已保存到: {final_save_path}")
        print("=" * 80)
        
        writer.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)

