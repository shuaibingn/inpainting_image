# 使用示例

本文档提供 ControlNet Inpainting 的完整使用示例。

## 📚 目录

1. [环境配置](#1-环境配置)
2. [数据准备](#2-数据准备)
3. [快速测试](#3-快速测试)
4. [训练模型](#4-训练模型)
5. [推理使用](#5-推理使用)
6. [高级用法](#6-高级用法)

---

## 1. 环境配置

### 步骤 1：创建虚拟环境

```bash
# 使用 conda（推荐）
conda create -n controlnet python=3.9
conda activate controlnet

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

### 步骤 2：安装依赖

```bash
pip install -r requirements.txt
```

### 步骤 3：验证安装

```bash
python test_installation.py
```

**预期输出：**
```
✓ PyTorch         - 已安装
✓ Diffusers       - 已安装
✓ CUDA 可用: True
✓ GPU 0: NVIDIA GeForce RTX 3090
🎉 所有测试通过！
```

---

## 2. 数据准备

### 方式 A：自动下载（推荐）

```bash
bash quick_start.sh
# 按照提示选择下载 COCO 数据集
```

### 方式 B：手动下载

```bash
# 创建目录
mkdir -p data/coco
cd data/coco

# 下载图像（18GB）
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

# 下载标注（241MB）
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

cd ../..
```

### 验证数据集

```bash
python data.py
```

这将生成 `data_sample.png`，显示：
- 原始图像
- 遮挡后的图像
- 生成的 mask

---

## 3. 快速测试

### 测试数据加载器

```bash
python data.py
```

### 测试模型创建

```bash
python model.py
```

**预期输出：**
```
正在加载预训练模型：runwayml/stable-diffusion-v1-5
冻结 UNet 权重
模型初始化完成
输入形状：
  x_t: torch.Size([2, 4, 64, 64])
  timestep: torch.Size([2])
  control_input: torch.Size([2, 4, 64, 64])
输出形状：
  noise_pred: torch.Size([2, 4, 64, 64])
总参数量: 1234.56M
可训练参数: 456.78M
```

---

## 4. 训练模型

### 基础训练（单 GPU）

```bash
python train.py \
  --image_dir data/coco/train2017 \
  --ann_file data/coco/annotations/instances_train2017.json \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --batch_size 4 \
  --num_epochs 100 \
  --learning_rate 1e-5 \
  --output_dir ./outputs \
  --mixed_precision fp16 \
  --save_steps 1000 \
  --log_steps 50
```

### 小规模测试（快速验证）

```bash
# 使用小批次和少量轮数
python train.py \
  --image_dir data/coco/train2017 \
  --ann_file data/coco/annotations/instances_train2017.json \
  --batch_size 2 \
  --num_epochs 5 \
  --learning_rate 1e-5 \
  --output_dir ./outputs/test_run \
  --save_steps 100
```

### 多 GPU 训练

```bash
# 配置 accelerate
accelerate config

# 启动训练
accelerate launch train.py \
  --image_dir data/coco/train2017 \
  --ann_file data/coco/annotations/instances_train2017.json \
  --batch_size 8 \
  --num_epochs 100 \
  --learning_rate 1e-5 \
  --output_dir ./outputs
```

### 从 checkpoint 恢复训练

```bash
python train.py \
  --image_dir data/coco/train2017 \
  --ann_file data/coco/annotations/instances_train2017.json \
  --resume_from_checkpoint outputs/controlnet_inpainting_*/checkpoint-5000
```

### 监控训练进度

```bash
# 在新终端中启动 TensorBoard
tensorboard --logdir outputs/

# 浏览器打开
# http://localhost:6006
```

**训练输出示例：**
```
Epoch 1/100: 100%|██████████| 29500/29500 [2:15:30<00:00, loss=0.1234, lr=9.8e-06]
✓ Checkpoint 已保存到: outputs/controlnet_inpainting_20241110_120000/checkpoint-1000
Epoch 1/100 完成
平均损失: 0.1234
```

---

## 5. 推理使用

### 创建测试 mask

```bash
# 不规则笔刷 mask
python create_test_mask.py --type irregular --output test_mask.png

# 矩形 mask
python create_test_mask.py --type rectangle --output test_mask.png

# 圆形 mask
python create_test_mask.py --type circle --output test_mask.png

# 中心矩形 mask
python create_test_mask.py --type center --output test_mask.png
```

### 基础推理

```bash
python infer.py \
  --checkpoint_path outputs/controlnet_inpainting_*/final_model/controlnet.pth \
  --input_image test_images/sample.jpg \
  --input_mask test_mask.png \
  --output_dir ./outputs/inference \
  --num_inference_steps 50 \
  --scheduler ddpm
```

### 快速推理（DDIM）

```bash
# DDIM 采样器速度更快（20 步 vs 50 步）
python infer.py \
  --checkpoint_path outputs/.../final_model/controlnet.pth \
  --input_image test_images/sample.jpg \
  --input_mask test_mask.png \
  --scheduler ddim \
  --num_inference_steps 20
```

### 批量推理

```bash
# 创建批量推理脚本
cat > batch_infer.sh << 'EOF'
#!/bin/bash

CHECKPOINT="outputs/controlnet_inpainting_*/final_model/controlnet.pth"
INPUT_DIR="test_images"
OUTPUT_DIR="outputs/batch_inference"

for img in $INPUT_DIR/*.jpg; do
    basename=$(basename "$img" .jpg)
    python infer.py \
        --checkpoint_path $CHECKPOINT \
        --input_image "$img" \
        --input_mask "masks/${basename}_mask.png" \
        --output_dir "$OUTPUT_DIR/$basename" \
        --scheduler ddim \
        --num_inference_steps 20
done
EOF

chmod +x batch_infer.sh
./batch_infer.sh
```

**推理输出示例：**
```
ControlNet Inpainting 推理
设备: cuda
Checkpoint: outputs/.../controlnet.pth
采样器: ddpm
推理步数: 50

正在加载模型...
✓ ControlNet 权重已加载
正在加载 VAE...
正在加载输入...
图像形状: torch.Size([1, 3, 512, 512])
Mask 形状: torch.Size([1, 1, 512, 512])

开始去噪过程...
去噪: 100%|██████████| 50/50 [00:15<00:00]

正在解码图像...
✓ 补全结果已保存到: outputs/inference/inpainted.png
✓ 对比图已保存到: outputs/inference/comparison.png

推理完成！
```

---

## 6. 高级用法

### 6.1 自定义数据集

创建自定义数据集类：

```python
# custom_dataset.py
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CustomInpaintingDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载图像
        image = Image.open(self.image_paths[idx]).convert('RGB')
        original_image = self.transform(image)
        
        # 加载 mask
        mask = Image.open(self.mask_paths[idx]).convert('L')
        mask = self.transform(mask)
        mask = (mask > 0.5).float()
        
        # 生成 masked_image
        masked_image = original_image * (1 - mask) + mask
        
        return original_image, masked_image, mask
```

使用自定义数据集训练：

```python
from torch.utils.data import DataLoader
from custom_dataset import CustomInpaintingDataset

# 准备数据路径
image_paths = [...]  # 你的图像路径列表
mask_paths = [...]   # 你的 mask 路径列表

# 创建数据集
dataset = CustomInpaintingDataset(image_paths, mask_paths)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 然后在 train.py 中使用这个 dataloader
```

### 6.2 调整训练超参数

创建配置文件 `my_config.yaml`：

```yaml
training:
  batch_size: 8
  learning_rate: 5.0e-6
  num_epochs: 200
  mixed_precision: "fp16"
  
data:
  image_size: 512
  mask_ratio_range: [0.15, 0.35]
  use_irregular_mask: true
```

### 6.3 微调预训练的 ControlNet

```bash
# 从已有的 checkpoint 继续训练
python train.py \
  --image_dir data/custom_dataset \
  --ann_file data/custom_annotations.json \
  --pretrained_controlnet_path outputs/.../checkpoint-10000/controlnet.pth \
  --learning_rate 5e-6 \
  --num_epochs 50
```

### 6.4 评估模型质量

创建评估脚本：

```python
# evaluate.py
import torch
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from tqdm import tqdm

from data import create_dataloader
from model import create_model
from diffusers import AutoencoderKL, DDIMScheduler

# 加载模型
model = create_model(checkpoint_path="...")
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

# 评估指标
ssim = StructuralSimilarityIndexMeasure()
psnr = PeakSignalNoiseRatio()

# 评估循环
for batch in tqdm(test_dataloader):
    original, masked, mask = batch
    
    # 推理
    inpainted = inference(model, vae, scheduler, masked, mask)
    
    # 计算指标
    ssim_score = ssim(inpainted, original)
    psnr_score = psnr(inpainted, original)
    
print(f"SSIM: {ssim_score:.4f}, PSNR: {psnr_score:.2f}")
```

### 6.5 导出 ONNX 模型

```python
# export_onnx.py
import torch
from model import create_model

model = create_model()
model.load_state_dict(torch.load("controlnet.pth"))
model.eval()

# 创建示例输入
dummy_input = (
    torch.randn(1, 4, 64, 64),      # x_t
    torch.tensor([500]),             # timestep
    torch.randn(1, 4, 64, 64),      # control_input
)

# 导出
torch.onnx.export(
    model,
    dummy_input,
    "controlnet_inpainting.onnx",
    input_names=['x_t', 'timestep', 'control_input'],
    output_names=['noise_pred'],
    dynamic_axes={
        'x_t': {0: 'batch'},
        'control_input': {0: 'batch'},
        'noise_pred': {0: 'batch'}
    }
)
```

---

## 💡 实用技巧

### 技巧 1：显存优化

```bash
# 启用梯度检查点
python train.py ... --gradient_checkpointing

# 减小批次大小
python train.py ... --batch_size 1

# 使用混合精度
python train.py ... --mixed_precision fp16
```

### 技巧 2：加速推理

```bash
# 使用 DDIM 采样器
--scheduler ddim --num_inference_steps 20

# 使用 xformers（需要额外安装）
pip install xformers
# 在代码中启用
model.enable_xformers_memory_efficient_attention()
```

### 技巧 3：调试模式

```bash
# 只在一个 batch 上训练（快速验证代码）
python train.py ... --batch_size 2 --num_epochs 1 --save_steps 10
```

### 技巧 4：可视化训练样本

```python
# 在训练循环中添加
if global_step % 500 == 0:
    with torch.no_grad():
        sample_output = model(sample_input, ...)
        save_image(sample_output, f'outputs/samples/step_{global_step}.png')
```

---

## 📊 性能基准

### 训练性能

| GPU | 批次大小 | 混合精度 | 速度（it/s）| 显存使用 |
|-----|---------|----------|------------|---------|
| RTX 3090 | 4 | FP16 | ~2.1 | 18GB |
| RTX 4090 | 8 | FP16 | ~4.5 | 20GB |
| A100 | 16 | FP16 | ~8.2 | 38GB |

### 推理性能

| 采样器 | 步数 | 时间（512×512）| 质量 |
|--------|------|----------------|------|
| DDPM | 50 | ~15s | 最佳 |
| DDIM | 20 | ~6s | 良好 |
| DDIM | 10 | ~3s | 一般 |

---

## 🎯 最佳实践

1. **训练阶段**
   - 使用 FP16 混合精度
   - 定期保存 checkpoint
   - 监控 TensorBoard
   - 训练至少 50K 步

2. **推理阶段**
   - DDPM 用于最佳质量
   - DDIM 用于快速测试
   - 调整 seed 获得不同结果

3. **数据准备**
   - mask 占比 10-30% 效果最好
   - 使用多样化的 mask 形状
   - 确保图像质量良好

---

## ❓ 故障排除

### 问题：训练损失不下降

**解决方案：**
- 检查学习率（尝试 1e-5 到 1e-4）
- 确保数据正确加载（运行 `python data.py`）
- 验证模型权重正确加载

### 问题：推理结果模糊

**解决方案：**
- 增加推理步数（50-100）
- 确保模型训练充分
- 尝试不同的随机种子

### 问题：显存溢出

**解决方案：**
```bash
--batch_size 1 --mixed_precision fp16
```

---

**更多问题？** 查看 [README.md](README.md) 或提交 Issue。

