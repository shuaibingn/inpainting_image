# PyTorch Lightning 版本使用指南

本文档介绍如何使用 PyTorch Lightning 版本的 ControlNet Inpainting。

## 🎯 为什么使用 Lightning？

PyTorch Lightning 版本提供以下优势：

### ✅ 代码更简洁
- 训练逻辑封装在 `LightningModule` 中
- 数据加载封装在 `LightningDataModule` 中
- 主训练脚本只需几行代码

### ✅ 自动化功能
- 自动处理分布式训练（DDP/DeepSpeed）
- 自动混合精度训练
- 自动梯度累积
- 自动 checkpoint 管理
- 自动日志记录

### ✅ 更好的可扩展性
- 易于添加新功能（callbacks）
- 易于切换训练策略
- 易于调试和测试

### ⚠️ 训练逻辑完全相同
Lightning 版本的训练逻辑与原始版本**完全相同**：
1. VAE 编码到 latent space
2. 采样时间步 t
3. 对 latent 加噪
4. ControlNet 预测噪声
5. 计算 MSE 损失
6. 反向传播更新参数

---

## 📂 文件结构

Lightning 版本新增以下文件：

```
inpainting_image/
├── lightning_module.py      # Lightning 模块（模型 + 训练逻辑）
├── lightning_data.py         # Lightning DataModule（数据加载）
├── train_lightning.py        # Lightning 训练脚本
└── LIGHTNING_GUIDE.md       # 本文档
```

**保留的文件**（逻辑不变）：
- `data.py` - 数据集类
- `model.py` - ControlNet 模型定义
- `infer.py` - 推理脚本（完全兼容）

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

确保已安装 `pytorch-lightning>=2.0.0`。

### 2. 基础训练

```bash
python train_lightning.py \
  --image_dir data/coco/train2017 \
  --ann_file data/coco/annotations/instances_train2017.json \
  --batch_size 4 \
  --num_epochs 100 \
  --learning_rate 1e-5
```

就这么简单！Lightning 会自动处理其他所有事情。

### 3. 监控训练

```bash
# 启动 TensorBoard
tensorboard --logdir outputs/

# 浏览器打开
# http://localhost:6006
```

---

## 💡 训练参数详解

### 基础参数

```bash
python train_lightning.py \
  --image_dir data/coco/train2017 \              # 图像目录
  --ann_file data/coco/annotations/instances_train2017.json \  # 标注文件
  --batch_size 4 \                                # 批次大小
  --num_epochs 100 \                              # 训练轮数
  --learning_rate 1e-5 \                          # 学习率
  --output_dir ./outputs                          # 输出目录
```

### Lightning 特有参数

```bash
# 混合精度训练
--precision 16-mixed    # FP16（推荐）
--precision bf16-mixed  # BF16（如果支持）
--precision 32          # FP32（精度最高但最慢）

# 分布式训练
--devices 2             # 使用 2 个 GPU
--strategy ddp          # 使用 DDP 策略

# 梯度累积（等效于更大的 batch size）
--accumulate_grad_batches 4  # 累积 4 个 batch

# Checkpoint 管理
--save_every_n_epochs 5      # 每 5 个 epoch 保存
--save_top_k 3               # 保存最好的 3 个 checkpoint

# 日志记录
--log_every_n_steps 50       # 每 50 步记录日志
--log_images_every_n_steps 500  # 每 500 步记录图像
```

---

## 🖥️ 单 GPU 训练

最简单的配置：

```bash
python train_lightning.py \
  --image_dir data/coco/train2017 \
  --ann_file data/coco/annotations/instances_train2017.json \
  --batch_size 4 \
  --num_epochs 100 \
  --precision 16-mixed \
  --devices 1
```

---

## 🔥 多 GPU 训练

### 方式 1：自动检测所有 GPU

```bash
python train_lightning.py \
  --image_dir data/coco/train2017 \
  --ann_file data/coco/annotations/instances_train2017.json \
  --batch_size 8 \
  --num_epochs 100 \
  --devices -1 \          # 使用所有可用 GPU
  --strategy ddp          # DDP 策略
```

### 方式 2：指定 GPU 数量

```bash
python train_lightning.py \
  --image_dir data/coco/train2017 \
  --ann_file data/coco/annotations/instances_train2017.json \
  --batch_size 8 \
  --num_epochs 100 \
  --devices 2 \           # 使用 2 个 GPU
  --strategy ddp
```

### 方式 3：使用 DeepSpeed（大规模训练）

```bash
python train_lightning.py \
  --image_dir data/coco/train2017 \
  --ann_file data/coco/annotations/instances_train2017.json \
  --batch_size 16 \
  --num_epochs 100 \
  --devices 4 \
  --strategy deepspeed    # DeepSpeed 策略
```

---

## 💾 Checkpoint 管理

### 自动保存

Lightning 会自动保存 checkpoint：

```
outputs/controlnet_inpainting_lightning_*/
├── checkpoints/
│   ├── controlnet-epoch=00-train_loss=0.1234.ckpt
│   ├── controlnet-epoch=05-train_loss=0.0987.ckpt
│   ├── controlnet-epoch=10-train_loss=0.0876.ckpt
│   └── last.ckpt                    # 最后一个 checkpoint
└── final_model/
    └── controlnet.pth               # 最终模型（仅 ControlNet 权重）
```

### 从 Checkpoint 恢复训练

```bash
python train_lightning.py \
  --image_dir data/coco/train2017 \
  --ann_file data/coco/annotations/instances_train2017.json \
  --resume_from_checkpoint outputs/.../checkpoints/last.ckpt
```

### 提取 ControlNet 权重用于推理

训练结束后，`final_model/controlnet.pth` 可以直接用于推理：

```bash
python infer.py \
  --checkpoint_path outputs/.../final_model/controlnet.pth \
  --input_image test.jpg \
  --input_mask mask.png
```

---

## 🎨 可视化和日志

### TensorBoard 日志

Lightning 自动记录以下信息：

- **损失曲线**：`train/loss`, `val/loss`
- **学习率**：`train/lr`
- **训练图像**：每 N 步记录一次
  - 原始图像
  - 遮挡图像
  - Mask

查看方式：

```bash
tensorboard --logdir outputs/
```

### 进度条

Lightning 提供详细的进度条信息：

```
Epoch 5: 100%|███████| 29500/29500 [2:15:30<00:00, loss=0.0987, lr=9.5e-06, v_num=0]
```

---

## 🔧 高级功能

### 1. 梯度累积（模拟更大 batch size）

如果显存不足，使用梯度累积：

```bash
python train_lightning.py \
  --batch_size 2 \                    # 实际 batch size
  --accumulate_grad_batches 4 \       # 累积 4 个 batch
  # 等效于 batch_size=8
```

### 2. 梯度裁剪

防止梯度爆炸：

```bash
python train_lightning.py \
  --gradient_clip_val 1.0    # 裁剪梯度范数到 1.0
```

### 3. 验证集评估

自动划分验证集：

```bash
python train_lightning.py \
  --val_split 0.05    # 5% 作为验证集
```

Lightning 会在每个 epoch 结束后自动运行验证。

### 4. 提前停止（Early Stopping）

可以添加 Early Stopping callback（需要修改代码）：

```python
from pytorch_lightning.callbacks import EarlyStopping

early_stop_callback = EarlyStopping(
    monitor='val/loss',
    patience=10,
    mode='min'
)
```

### 5. 学习率查找

Lightning 提供自动学习率查找：

```python
# 在 train_lightning.py 中添加
trainer = pl.Trainer(auto_lr_find=True, ...)
trainer.tune(model, datamodule)  # 自动找最佳学习率
```

---

## 📊 性能对比

| 特性 | 原始版本 | Lightning 版本 |
|-----|---------|---------------|
| **代码行数** | ~420 行 | ~150 行 |
| **训练速度** | 相同 | 相同 |
| **显存使用** | 相同 | 相同 |
| **分布式训练** | 需要手动配置 | 自动处理 |
| **混合精度** | 需要 Accelerate | 内置支持 |
| **Checkpoint** | 手动管理 | 自动管理 |
| **日志记录** | 手动实现 | 自动记录 |
| **可维护性** | 中等 | 高 |

---

## 🆚 两个版本的选择

### 使用原始版本（train.py）如果：
- ✅ 需要完全控制训练流程
- ✅ 想了解每个细节的实现
- ✅ 需要自定义非标准的训练逻辑

### 使用 Lightning 版本（train_lightning.py）如果：
- ✅ 想要更简洁的代码
- ✅ 需要分布式训练
- ✅ 想要自动化的功能
- ✅ 注重代码可维护性

**推荐：** 新项目使用 Lightning 版本，学习目的使用原始版本。

---

## 🔄 迁移指南

### 从原始版本迁移到 Lightning

如果你已经使用原始版本训练了一些 checkpoint：

1. **权重兼容**：Lightning 版本与原始版本的模型权重完全兼容

2. **继续训练**：可以从原始版本的 checkpoint 继续训练
   ```bash
   # 提取 ControlNet 权重
   python -c "
   import torch
   ckpt = torch.load('outputs/old/checkpoint-1000/controlnet.pth')
   # 在 Lightning 中加载
   "
   ```

3. **推理兼容**：推理脚本 `infer.py` 对两个版本完全兼容

---

## 💡 实用技巧

### 技巧 1：快速验证代码

```bash
# 使用小数据集快速测试
python train_lightning.py \
  --batch_size 2 \
  --num_epochs 1 \
  --max_steps 100 \
  --log_every_n_steps 10
```

### 技巧 2：过拟合单个 batch（调试）

在 Lightning 中添加：

```python
trainer = pl.Trainer(
    overfit_batches=1,  # 只在 1 个 batch 上训练
    ...
)
```

### 技巧 3：性能分析

```python
trainer = pl.Trainer(
    profiler="simple",  # 或 "advanced"
    ...
)
```

### 技巧 4：确定性训练

```bash
python train_lightning.py \
  --seed 42              # 固定随机种子
```

Lightning 会自动设置所有随机种子。

---

## ❓ 常见问题

### Q1: Lightning 版本会影响训练结果吗？

**A:** 不会。训练逻辑完全相同，只是代码组织方式不同。

### Q2: 能否在训练中途切换版本？

**A:** 可以。模型权重完全兼容，可以互相转换。

### Q3: Lightning 版本的显存使用会更多吗？

**A:** 不会。显存使用与原始版本相同。

### Q4: 如何调试 Lightning 代码？

**A:** 使用 `--devices 1` 和 `--num_workers 0` 可以方便调试。

### Q5: 支持 TPU 训练吗？

**A:** 支持。使用 `--accelerator tpu` 即可。

---

## 📚 更多资源

- **PyTorch Lightning 官方文档**：https://lightning.ai/docs/pytorch/
- **Lightning 示例**：https://github.com/Lightning-AI/lightning
- **ControlNet 论文**：https://arxiv.org/abs/2302.05543

---

## 🎯 快速命令参考

```bash
# 单 GPU 训练
python train_lightning.py --image_dir data/coco/train2017 --ann_file data/coco/annotations/instances_train2017.json --batch_size 4 --num_epochs 100

# 多 GPU 训练
python train_lightning.py --image_dir data/coco/train2017 --ann_file data/coco/annotations/instances_train2017.json --batch_size 8 --devices 2 --strategy ddp

# 从 checkpoint 恢复
python train_lightning.py --image_dir data/coco/train2017 --ann_file data/coco/annotations/instances_train2017.json --resume_from_checkpoint outputs/.../checkpoints/last.ckpt

# 推理（完全兼容）
python infer.py --checkpoint_path outputs/.../final_model/controlnet.pth --input_image test.jpg --input_mask mask.png

# 查看日志
tensorboard --logdir outputs/
```

---

**享受更简洁的训练体验！** ⚡

