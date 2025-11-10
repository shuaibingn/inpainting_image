"""
ControlNet Inpainting 训练脚本（PyTorch Lightning 版本）

使用 PyTorch Lightning 重写的训练脚本，保持原有训练逻辑不变。

优势：
- 更简洁的代码组织
- 自动处理分布式训练
- 内置的 checkpoint 管理
- 更好的日志记录
- 自动的梯度累积和混合精度

训练逻辑完全相同：
1. VAE 编码到 latent space
2. 采样时间步 t
3. 对 latent 加噪
4. ControlNet 前向传播预测噪声
5. 计算 MSE 损失
6. 反向传播更新参数
"""

import os
import argparse
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from lightning_module import ControlNetInpaintingLightningModule
from lightning_data import ControlNetInpaintingDataModule


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练 ControlNet Inpainting 模型（Lightning 版本）")
    
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
        "--max_steps",
        type=int,
        default=-1,
        help="最大训练步数（-1 表示使用 num_epochs）"
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
        "--save_every_n_epochs",
        type=int,
        default=5,
        help="每隔多少 epoch 保存一次 checkpoint"
    )
    parser.add_argument(
        "--save_top_k",
        type=int,
        default=3,
        help="保存最好的 k 个 checkpoint"
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=50,
        help="每隔多少步记录一次日志"
    )
    parser.add_argument(
        "--log_images_every_n_steps",
        type=int,
        default=500,
        help="每隔多少步记录一次图像"
    )
    
    # Lightning 参数
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        choices=["32", "16-mixed", "bf16-mixed"],
        help="训练精度（32/16-mixed/bf16-mixed）"
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "gpu", "cpu", "tpu"],
        help="加速器类型"
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="使用的设备数量"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        choices=["auto", "ddp", "ddp_spawn", "deepspeed"],
        help="分布式训练策略"
    )
    parser.add_argument(
        "--gradient_clip_val",
        type=float,
        default=1.0,
        help="梯度裁剪值"
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="梯度累积批次数"
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
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.05,
        help="验证集比例（0-1）"
    )
    
    # 恢复训练
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="从 checkpoint 恢复训练"
    )
    
    args = parser.parse_args()
    return args


def main():
    """主训练函数"""
    args = parse_args()
    
    print("=" * 80)
    print("ControlNet Inpainting 训练（PyTorch Lightning）")
    print("=" * 80)
    
    # 设置随机种子
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"controlnet_inpainting_lightning_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"输出目录: {output_dir}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"精度: {args.precision}")
    print("=" * 80)
    
    # 创建 DataModule
    print("\n正在创建数据模块...")
    datamodule = ControlNetInpaintingDataModule(
        image_dir=args.image_dir,
        ann_file=args.ann_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_irregular_mask=args.use_irregular_mask,
        val_split=args.val_split,
    )
    
    # 创建 Lightning 模块
    print("正在创建模型...")
    model = ControlNetInpaintingLightningModule(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        learning_rate=args.learning_rate,
        adam_weight_decay=args.adam_weight_decay,
        freeze_unet=True,
        log_images_every_n_steps=args.log_images_every_n_steps,
    )
    
    # 创建 TensorBoard Logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="logs",
        version="",
    )
    
    # 创建 Callbacks
    callbacks = []
    
    # Checkpoint 回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="controlnet-{epoch:02d}-{train/loss:.4f}",
        save_top_k=args.save_top_k,
        monitor="train/loss",
        mode="min",
        every_n_epochs=args.save_every_n_epochs,
        save_last=True,  # 总是保存最后一个 checkpoint
    )
    callbacks.append(checkpoint_callback)
    
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # 进度条
    progress_bar = TQDMProgressBar(refresh_rate=10)
    callbacks.append(progress_bar)
    
    # 创建 Trainer
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy if args.devices > 1 else "auto",
        precision=args.precision,
        max_epochs=args.num_epochs,
        max_steps=args.max_steps,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        deterministic=True if args.seed is not None else False,
        # 性能优化
        benchmark=True,  # cuDNN benchmark
        enable_model_summary=True,
    )
    
    # 打印模型信息
    print("\n模型信息：")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params / 1e6:.2f}M")
    print(f"  可训练参数: {trainable_params / 1e6:.2f}M")
    print(f"  冻结参数: {(total_params - trainable_params) / 1e6:.2f}M")
    
    # 开始训练
    print("\n" + "=" * 80)
    print("开始训练...")
    print("=" * 80 + "\n")
    
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=args.resume_from_checkpoint,
    )
    
    # 训练完成
    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    
    # 保存最终模型
    print("\n保存最终模型...")
    final_model_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    
    # 只保存 ControlNet 权重
    import torch
    torch.save(
        model.model.controlnet.state_dict(),
        os.path.join(final_model_path, "controlnet.pth")
    )
    
    print(f"✓ 最终模型已保存到: {final_model_path}")
    print(f"✓ Checkpoints 保存在: {os.path.join(output_dir, 'checkpoints')}")
    print(f"✓ TensorBoard 日志: {os.path.join(output_dir, 'logs')}")
    print("\n查看训练日志：")
    print(f"  tensorboard --logdir {output_dir}/logs")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

