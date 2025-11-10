# 项目结构说明

本文档说明 ControlNet Inpainting 项目的文件组织结构。

## 📁 文件结构

```
inpainting_image/
│
├── 📄 核心代码文件
│   ├── data.py                    # 数据集加载器
│   ├── model.py                   # ControlNet 模型定义
│   ├── train.py                   # 训练脚本
│   └── infer.py                   # 推理脚本
│
├── 📄 配置和文档
│   ├── requirements.txt           # Python 依赖
│   ├── config_example.yaml        # 配置文件示例
│   ├── README.md                  # 主文档
│   ├── PROJECT_STRUCTURE.md       # 本文件
│   └── .gitignore                 # Git 忽略规则
│
├── 📄 辅助工具
│   ├── create_test_mask.py        # 创建测试 mask
│   ├── test_installation.py       # 测试环境安装
│   └── quick_start.sh             # 快速启动脚本
│
└── 📁 数据和输出（运行时生成）
    ├── data/                      # 数据集目录
    │   └── coco/
    │       ├── train2017/         # COCO 训练图像
    │       └── annotations/       # COCO 标注文件
    │
    └── outputs/                   # 训练输出
        └── controlnet_inpainting_*/
            ├── logs/              # TensorBoard 日志
            ├── checkpoint-*/      # 训练检查点
            └── final_model/       # 最终模型
```

## 📝 核心文件说明

### 1. data.py - 数据集加载器

**功能：**
- 加载 COCO 数据集
- 自动生成随机 mask（矩形/不规则）
- 创建 masked_image
- 数据预处理和增强

**关键类：**
- `CocoInpaintingDataset`: 主数据集类
- `create_dataloader()`: 便捷的数据加载器创建函数

**输出维度：**
- `original_image`: [3, 512, 512] - 原始图像
- `masked_image`: [3, 512, 512] - 遮挡后的图像
- `mask`: [1, 512, 512] - 二值掩码

**测试方法：**
```bash
python data.py
```

---

### 2. model.py - ControlNet 模型

**功能：**
- 定义 ControlNet 架构
- 实现 Zero Convolution
- 实现特征注入机制
- 加载预训练权重

**关键类：**
- `ZeroConv`: 零初始化卷积层
- `ControlNetConditioningEmbedding`: 条件编码模块
- `ControlNetModel`: ControlNet 主体
- `ControlNetInpaintingModel`: 完整模型（UNet + ControlNet）

**模型输入：**
- `x_t`: [B, 4, 64, 64] - 加噪的 latent
- `timestep`: [B] - 时间步
- `control_input`: [B, 4, 64, 64] - 控制输入
- `encoder_hidden_states`: [B, seq_len, dim] - 文本编码（可选）

**模型输出：**
- `noise_pred`: [B, 4, 64, 64] - 预测的噪声

**测试方法：**
```bash
python model.py
```

---

### 3. train.py - 训练脚本

**功能：**
- 完整的训练循环
- VAE 编码/解码
- 混合精度训练
- 分布式训练支持
- Checkpoint 保存
- TensorBoard 日志

**训练流程：**
1. 加载数据
2. 初始化模型（加载预训练权重）
3. 对每个 batch：
   - VAE 编码到 latent space
   - 采样时间步 t
   - 对 latent 加噪
   - ControlNet 前向传播
   - 计算 MSE 损失
   - 反向传播更新参数
4. 定期保存 checkpoint

**使用方法：**
```bash
python train.py \
  --image_dir data/coco/train2017 \
  --ann_file data/coco/annotations/instances_train2017.json \
  --batch_size 4 \
  --num_epochs 100
```

**输出文件：**
- `checkpoint-*/controlnet.pth`: ControlNet 权重
- `checkpoint-*/optimizer.pth`: 优化器状态
- `checkpoint-*/training_state.pth`: 训练状态
- `logs/`: TensorBoard 日志

---

### 4. infer.py - 推理脚本

**功能：**
- 加载训练好的模型
- 图像补全推理
- 支持 DDPM/DDIM 采样器
- 生成对比图

**推理流程：**
1. 加载 ControlNet 权重
2. 加载输入图像和 mask
3. VAE 编码到 latent space
4. 初始化随机噪声
5. 逐步去噪（DDPM/DDIM）
6. VAE 解码到图像空间
7. 保存结果

**使用方法：**
```bash
python infer.py \
  --checkpoint_path outputs/.../controlnet.pth \
  --input_image test.jpg \
  --input_mask mask.png \
  --scheduler ddpm \
  --num_inference_steps 50
```

**输出文件：**
- `inpainted.png`: 补全后的图像
- `comparison.png`: 对比图

---

## 🛠️ 辅助工具

### create_test_mask.py

创建各种形状的测试 mask。

**支持的 mask 类型：**
- `rectangle`: 矩形
- `circle`: 圆形
- `irregular`: 不规则笔刷线条
- `center`: 中心矩形
- `text`: 文字形状

**使用方法：**
```bash
python create_test_mask.py --type irregular --output mask.png
```

---

### test_installation.py

测试环境是否正确配置。

**测试内容：**
1. 依赖库导入
2. PyTorch 和 CUDA 配置
3. Diffusers 库功能
4. 项目文件完整性
5. 模型创建

**使用方法：**
```bash
python test_installation.py
```

---

### quick_start.sh

一键式启动脚本，引导完成：
1. 环境检查
2. 依赖安装
3. 数据集下载
4. 数据加载器测试
5. 开始训练

**使用方法：**
```bash
chmod +x quick_start.sh
./quick_start.sh
```

---

## 📊 数据流程

### 训练数据流

```
原始图像 (512x512)
    ↓
生成 mask
    ↓
创建 masked_image
    ↓
[original_image, masked_image, mask]
    ↓
VAE Encoder
    ↓
latent (64x64)
    ↓
加噪
    ↓
ControlNet + UNet
    ↓
预测噪声
    ↓
计算损失
    ↓
反向传播
```

### 推理数据流

```
[masked_image, mask] (512x512)
    ↓
VAE Encoder
    ↓
latent (64x64)
    ↓
随机噪声
    ↓
循环去噪 (T → 0)
│   ├── ControlNet + UNet
│   └── 更新 latent
    ↓
最终 latent
    ↓
VAE Decoder
    ↓
补全图像 (512x512)
```

---

## 🎯 快速开始指南

### 1. 测试环境
```bash
python test_installation.py
```

### 2. 测试数据加载
```bash
python data.py
```

### 3. 测试模型
```bash
python model.py
```

### 4. 开始训练
```bash
python train.py \
  --image_dir data/coco/train2017 \
  --ann_file data/coco/annotations/instances_train2017.json \
  --batch_size 4 \
  --num_epochs 100
```

### 5. 进行推理
```bash
# 创建测试 mask
python create_test_mask.py --type irregular --output test_mask.png

# 运行推理
python infer.py \
  --checkpoint_path outputs/.../final_model/controlnet.pth \
  --input_image test.jpg \
  --input_mask test_mask.png
```

---

## 📦 依赖关系

```
requirements.txt
    ├── torch (核心深度学习框架)
    ├── diffusers (Stable Diffusion 和调度器)
    ├── transformers (文本编码器)
    ├── accelerate (分布式训练)
    ├── pycocotools (COCO 数据集)
    └── PIL, opencv, numpy (图像处理)
```

---

## 💾 存储需求

### 模型权重
- Stable Diffusion v1.5: ~5GB
- ControlNet: ~1.5GB
- VAE: ~335MB

### 数据集
- COCO train2017: ~18GB
- COCO annotations: ~241MB

### 训练输出
- 每个 checkpoint: ~1.5GB
- TensorBoard 日志: ~100MB

**总计建议存储空间：50GB+**

---

## 🔍 常见问题排查

### 问题 1: ModuleNotFoundError
**解决：** 运行 `pip install -r requirements.txt`

### 问题 2: CUDA out of memory
**解决：** 减小 `--batch_size` 或使用 `--mixed_precision fp16`

### 问题 3: 数据集加载失败
**解决：** 检查路径是否正确，运行 `python data.py` 测试

### 问题 4: 模型下载慢
**解决：** 设置 HuggingFace 镜像或提前下载模型

---

## 📞 获取帮助

- 查看 README.md 了解详细使用说明
- 运行 `python [script].py --help` 查看参数说明
- 查看代码注释了解实现细节

---

**祝使用愉快！** 🚀

