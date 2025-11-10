# ControlNet Inpainting - 图像补全模型

基于 [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet) 实现的图像补全（Inpainting）模型。

本项目使用 **Stable Diffusion v1.5** 作为基础模型，训练 ControlNet 来学习根据遮挡图像和 mask 进行图像补全。

## 📋 目录

- [特性](#特性)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [训练](#训练)
- [推理](#推理)
- [项目结构](#项目结构)
- [技术细节](#技术细节)
- [常见问题](#常见问题)

## ✨ 特性

- ✅ 基于 lllyasviel/ControlNet 原始实现
- ✅ 使用 Stable Diffusion v1.5 预训练权重
- ✅ 自动从 COCO 数据集生成训练数据
- ✅ 支持矩形和不规则 mask
- ✅ 支持混合精度训练（FP16/BF16）
- ✅ 支持分布式训练（via Accelerate 或 PyTorch Lightning）
- ✅ 提供 DDPM 和 DDIM 采样器
- ✅ 完整的训练和推理流程
- ✅ TensorBoard 日志记录
- ✅ **新增：PyTorch Lightning 版本（更简洁、更强大）**

## 🔧 环境配置

### 1. 创建虚拟环境

```bash
# 使用 conda
conda create -n controlnet_inpainting python=3.9
conda activate controlnet_inpainting

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 主要依赖

- PyTorch >= 2.0
- Diffusers >= 0.21.0
- Transformers >= 4.30.0
- Accelerate >= 0.20.0
- CUDA 11.8+ (GPU 训练)

## 📦 数据准备

### 方式 1：使用 COCO 数据集（推荐）

1. **下载 COCO 数据集**

```bash
# 创建数据目录
mkdir -p data/coco

# 下载 train2017 图像集（~18GB）
cd data/coco
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

# 下载标注文件（~241MB）
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

2. **数据集结构**

```
data/coco/
├── train2017/           # 训练图像
│   ├── 000000000009.jpg
│   ├── 000000000025.jpg
│   └── ...
└── annotations/
    └── instances_train2017.json
```

### 方式 2：使用自定义数据集

修改 `data.py` 中的 `CocoInpaintingDataset` 类来适配你的数据格式。

### 测试数据加载器

```bash
# 修改 data.py 末尾的路径
python data.py
```

这将生成一个样本可视化图像 `data_sample.png`。

## 🚀 训练

本项目提供两个训练版本，训练逻辑完全相同：

### 版本 1：PyTorch Lightning（推荐 ⚡）

**更简洁、更强大、自动化程度更高**

```bash
python train_lightning.py \
  --image_dir data/coco/train2017 \
  --ann_file data/coco/annotations/instances_train2017.json \
  --batch_size 4 \
  --num_epochs 100 \
  --learning_rate 1e-5 \
  --precision 16-mixed \
  --devices 1
```

**优势：**
- ✨ 代码更简洁（150 行 vs 420 行）
- ✨ 自动分布式训练
- ✨ 自动 checkpoint 管理
- ✨ 更好的日志和可视化

详见 [LIGHTNING_GUIDE.md](LIGHTNING_GUIDE.md)

### 版本 2：原始训练脚本

**完整控制训练流程，适合学习**

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

### 训练参数说明

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `--image_dir` | COCO 图像目录 | 必填 |
| `--ann_file` | COCO 标注文件 | 必填 |
| `--pretrained_model_name_or_path` | Stable Diffusion 模型路径 | `runwayml/stable-diffusion-v1-5` |
| `--batch_size` | 批次大小 | 4 |
| `--num_epochs` | 训练轮数 | 100 |
| `--learning_rate` | 学习率 | 1e-5 |
| `--output_dir` | 输出目录 | `./outputs` |
| `--mixed_precision` | 混合精度（no/fp16/bf16） | fp16 |
| `--save_steps` | 保存间隔步数 | 1000 |
| `--num_workers` | 数据加载线程数 | 4 |

### 多GPU训练

```bash
# 使用 accelerate 配置多 GPU
accelerate config

# 启动训练
accelerate launch train.py \
  --image_dir data/coco/train2017 \
  --ann_file data/coco/annotations/instances_train2017.json \
  --batch_size 8 \
  --num_epochs 100
```

### 监控训练

```bash
# 启动 TensorBoard
tensorboard --logdir outputs/controlnet_inpainting_*/logs

# 在浏览器中打开
# http://localhost:6006
```

### 训练输出

训练过程中会生成以下文件：

```
outputs/controlnet_inpainting_20241110_120000/
├── logs/                          # TensorBoard 日志
├── checkpoint-1000/               # 中间 checkpoint
│   ├── controlnet.pth            # ControlNet 权重
│   ├── optimizer.pth             # 优化器状态
│   └── training_state.pth        # 训练状态
└── final_model/                   # 最终模型
    └── controlnet.pth
```

## 🎨 推理

### 基础推理命令

```bash
python infer.py \
  --checkpoint_path outputs/controlnet_inpainting_*/final_model/controlnet.pth \
  --input_image test_images/sample.jpg \
  --input_mask test_images/mask.png \
  --output_dir ./outputs/inference \
  --num_inference_steps 50 \
  --scheduler ddpm
```

### 推理参数说明

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `--checkpoint_path` | ControlNet 权重路径 | 必填 |
| `--input_image` | 输入图像路径 | 必填 |
| `--input_mask` | Mask 图像路径（白色=补全区域） | 必填 |
| `--output_dir` | 输出目录 | `./outputs/inference` |
| `--num_inference_steps` | 推理步数 | 50 |
| `--scheduler` | 采样器（ddpm/ddim） | ddpm |
| `--seed` | 随机种子 | 42 |

### 准备推理输入

1. **输入图像**：任意 RGB 图像
2. **Mask 图像**：灰度图像，白色区域表示需要补全的部分

```python
# 示例：创建简单的矩形 mask
from PIL import Image, ImageDraw

mask = Image.new('L', (512, 512), 0)
draw = ImageDraw.Draw(mask)
draw.rectangle([100, 100, 400, 400], fill=255)
mask.save('mask.png')
```

### 推理输出

```
outputs/inference/
├── inpainted.png      # 补全后的图像
└── comparison.png     # 对比图（输入/mask/输出）
```

### 快速推理（DDIM）

```bash
# 使用 DDIM 采样器，更快速（20 步）
python infer.py \
  --checkpoint_path outputs/.../final_model/controlnet.pth \
  --input_image test.jpg \
  --input_mask mask.png \
  --scheduler ddim \
  --num_inference_steps 20
```

## 📁 项目结构

```
inpainting_image/
├── 核心代码
│   ├── data.py                  # 数据集加载器
│   ├── model.py                 # ControlNet 模型定义
│   ├── train.py                 # 训练脚本（原始版本）
│   ├── train_lightning.py       # 训练脚本（Lightning 版本）⚡
│   ├── lightning_module.py      # Lightning 模块封装
│   ├── lightning_data.py        # Lightning DataModule
│   └── infer.py                 # 推理脚本
│
├── 文档
│   ├── README.md                # 主文档
│   ├── LIGHTNING_GUIDE.md       # Lightning 使用指南
│   ├── PROJECT_STRUCTURE.md     # 项目结构说明
│   └── USAGE_EXAMPLES.md        # 使用示例
│
├── 工具
│   ├── create_test_mask.py      # 创建测试 mask
│   ├── test_installation.py     # 测试环境
│   └── quick_start.sh           # 快速启动
│
└── 配置
    ├── requirements.txt         # 依赖库
    ├── config_example.yaml      # 配置示例
    └── .gitignore              # Git 忽略规则
```

## 🔬 技术细节

### 模型架构

```
输入: masked_image [B,3,512,512] + mask [B,1,512,512]
  ↓
VAE Encoder (冻结)
  ↓
latent [B,4,64,64]
  ↓
┌─────────────────────────────┐
│   ControlNet 分支            │
│   ├── Conditioning Embedding │
│   ├── Down Blocks           │
│   └── Zero Convolutions     │
└─────────────────────────────┘
  ↓ (特征注入)
┌─────────────────────────────┐
│   UNet 主干 (冻结)           │
│   ├── Down Blocks           │
│   ├── Mid Block             │
│   └── Up Blocks             │
└─────────────────────────────┘
  ↓
预测噪声 [B,4,64,64]
  ↓
VAE Decoder (冻结)
  ↓
输出: inpainted_image [B,3,512,512]
```

### 训练策略

1. **冻结 UNet**：只训练 ControlNet 分支，保持预训练权重
2. **Zero Convolution**：特征注入层初始化为零，训练初期不影响主干
3. **Latent Space**：在 64×64 latent space 中训练（不是 512×512）
4. **噪声预测**：预测加入的噪声，而非直接预测图像

### 关键超参数

- **学习率**：1e-5（AdamW）
- **训练步数**：建议 50K-100K 步
- **批次大小**：4-8（取决于 GPU 显存）
- **图像尺寸**：512×512（Stable Diffusion 标准）
- **Mask 比例**：10%-30% 图像面积

## ❓ 常见问题

### Q1: 训练需要多少显存？

- **最小配置**：12GB（batch_size=1, fp16）
- **推荐配置**：24GB（batch_size=4, fp16）
- **高配置**：40GB+（batch_size=8+）

### Q2: 训练需要多久？

- **单卡 RTX 3090**：约 3-5 天（100 epochs, 118K 图像）
- **多卡 A100**：约 1-2 天

### Q3: 首次运行很慢？

首次运行会自动下载预训练模型（~5GB），请耐心等待。可以设置 HuggingFace 缓存路径：

```bash
export HF_HOME=/path/to/cache
```

### Q4: 显存不足怎么办？

```bash
# 减小批次大小
--batch_size 1

# 使用混合精度
--mixed_precision fp16

# 减少数据加载线程
--num_workers 0
```

### Q5: 如何在自己的数据集上训练？

修改 `data.py` 中的 `CocoInpaintingDataset` 类：

```python
class CustomInpaintingDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
    
    def __getitem__(self, idx):
        # 加载你的数据
        image = load_image(self.image_paths[idx])
        mask = load_mask(self.mask_paths[idx])
        # 返回 original_image, masked_image, mask
        ...
```

### Q6: 推理结果不理想？

- 增加推理步数（`--num_inference_steps 100`）
- 尝试不同的随机种子
- 确保模型训练充分（至少 50K 步）
- 检查 mask 格式是否正确（白色=补全区域）

### Q7: 支持更高分辨率吗？

当前实现针对 512×512 优化。如需更高分辨率：

1. 使用 Stable Diffusion 2.x（支持 768×768）
2. 修改 `--image_size` 参数
3. 需要更多显存

## 📄 许可证

本项目基于 MIT 许可证开源。

## 🙏 致谢

- [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet) - 原始 ControlNet 实现
- [Stability AI](https://stability.ai/) - Stable Diffusion 模型
- [Hugging Face](https://huggingface.co/) - Diffusers 库

## 📧 联系方式

如有问题或建议，欢迎提 Issue 或 Pull Request。

---

**祝训练顺利！🎉**

