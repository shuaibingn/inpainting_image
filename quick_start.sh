#!/bin/bash

# ControlNet Inpainting 快速启动脚本
# 本脚本帮助你快速开始训练和推理

set -e

echo "=========================================="
echo "ControlNet Inpainting 快速启动"
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 检查环境
echo -e "\n${YELLOW}[1/5] 检查环境...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}错误: 未找到 Python${NC}"
    exit 1
fi

python_version=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python 版本: $python_version${NC}"

# 2. 安装依赖
echo -e "\n${YELLOW}[2/5] 检查依赖...${NC}"
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}错误: 未找到 requirements.txt${NC}"
    exit 1
fi

echo "是否安装依赖？(y/n)"
read -r install_deps
if [ "$install_deps" = "y" ]; then
    echo "正在安装依赖..."
    pip install -r requirements.txt
    echo -e "${GREEN}✓ 依赖安装完成${NC}"
else
    echo "跳过依赖安装"
fi

# 3. 下载 COCO 数据集（可选）
echo -e "\n${YELLOW}[3/5] 数据集准备${NC}"
echo "是否下载 COCO train2017 数据集？(y/n)"
echo "注意: 数据集约 18GB，下载需要较长时间"
read -r download_coco

if [ "$download_coco" = "y" ]; then
    echo "正在创建数据目录..."
    mkdir -p data/coco
    cd data/coco
    
    if [ ! -f "train2017.zip" ]; then
        echo "正在下载图像数据..."
        wget http://images.cocodataset.org/zips/train2017.zip
        unzip train2017.zip
        rm train2017.zip
    else
        echo "图像数据已存在，跳过下载"
    fi
    
    if [ ! -f "annotations_trainval2017.zip" ]; then
        echo "正在下载标注文件..."
        wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
        unzip annotations_trainval2017.zip
        rm annotations_trainval2017.zip
    else
        echo "标注文件已存在，跳过下载"
    fi
    
    cd ../..
    echo -e "${GREEN}✓ COCO 数据集准备完成${NC}"
else
    echo "跳过数据集下载"
    echo "请手动准备数据集，或修改 data.py 使用自定义数据"
fi

# 4. 测试数据加载器
echo -e "\n${YELLOW}[4/5] 测试数据加载器${NC}"
echo "是否测试数据加载器？(y/n)"
read -r test_data

if [ "$test_data" = "y" ]; then
    if [ ! -d "data/coco/train2017" ]; then
        echo -e "${RED}错误: 未找到 COCO 数据集${NC}"
        echo "请先下载数据集或修改 data.py 中的路径"
    else
        echo "正在测试数据加载器..."
        python data.py
        echo -e "${GREEN}✓ 数据加载器测试完成${NC}"
        echo "查看 data_sample.png 确认数据加载是否正确"
    fi
else
    echo "跳过数据加载器测试"
fi

# 5. 开始训练
echo -e "\n${YELLOW}[5/5] 训练模型${NC}"
echo "是否开始训练？(y/n)"
echo "注意: 训练需要 GPU 和较长时间"
read -r start_training

if [ "$start_training" = "y" ]; then
    echo "正在启动训练..."
    echo "使用默认参数："
    echo "  - 批次大小: 4"
    echo "  - 训练轮数: 100"
    echo "  - 学习率: 1e-5"
    echo "  - 混合精度: fp16"
    
    if [ ! -d "data/coco/train2017" ]; then
        echo -e "${RED}错误: 未找到 COCO 数据集${NC}"
        exit 1
    fi
    
    python train.py \
        --image_dir data/coco/train2017 \
        --ann_file data/coco/annotations/instances_train2017.json \
        --batch_size 4 \
        --num_epochs 100 \
        --learning_rate 1e-5 \
        --output_dir ./outputs \
        --mixed_precision fp16 \
        --save_steps 1000 \
        --log_steps 50
    
    echo -e "${GREEN}✓ 训练完成${NC}"
else
    echo "跳过训练"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}快速启动完成！${NC}"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 如果已完成训练，使用以下命令进行推理："
echo "   python infer.py --checkpoint_path outputs/.../final_model/controlnet.pth \\"
echo "                   --input_image test.jpg \\"
echo "                   --input_mask mask.png"
echo ""
echo "2. 监控训练进度："
echo "   tensorboard --logdir outputs/"
echo ""
echo "3. 查看完整文档："
echo "   cat README.md"
echo ""

