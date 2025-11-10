"""
COCO 数据集加载器 - 用于图像补全（Inpainting）任务

本模块提供：
- CocoInpaintingDataset: COCO 数据集类，自动生成 mask 和 masked_image
- 支持随机矩形和不规则形状的 mask 生成
- 数据增强和预处理

输入输出维度：
- original_image: [3, 512, 512] - 原始图像
- masked_image: [3, 512, 512] - 遮挡后的图像
- mask: [1, 512, 512] - 二值掩码，1 表示需要补全的区域
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO


class CocoInpaintingDataset(Dataset):
    """
    COCO 图像补全数据集类。
    
    自动生成随机 mask 用于图像补全训练。
    支持矩形和不规则形状的 mask。
    """
    
    def __init__(
        self, 
        image_dir, 
        ann_file, 
        image_size=512,
        mask_ratio_range=(0.1, 0.3),
        transform=None,
        use_irregular_mask=True
    ):
        """
        初始化数据集。
        
        参数：
        - image_dir: str, 图像文件夹路径
        - ann_file: str, COCO 标注文件路径
        - image_size: int, 输出图像尺寸（默认 512）
        - mask_ratio_range: tuple, mask 占图像面积的比例范围（默认 0.1-0.3）
        - transform: 自定义图像变换（可选）
        - use_irregular_mask: bool, 是否使用不规则 mask（默认 True）
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.mask_ratio_range = mask_ratio_range
        self.use_irregular_mask = use_irregular_mask
        
        # 加载 COCO 数据集
        print(f"正在加载 COCO 数据集：{ann_file}")
        self.coco = COCO(ann_file)
        self.image_ids = list(self.coco.imgs.keys())
        print(f"数据集加载完成，共 {len(self.image_ids)} 张图像")
        
        # 图像预处理变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到 [-1, 1]
            ])
        else:
            self.transform = transform
            
        # Mask 变换（不需要归一化）
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """
        获取数据集中的一个样本。
        
        参数：
        - idx: int, 样本索引
        
        返回：
        - original_image: tensor [3, 512, 512], 原始图像（归一化到 [-1, 1]）
        - masked_image: tensor [3, 512, 512], 遮挡后的图像
        - mask: tensor [1, 512, 512], 二值掩码（0-1）
        """
        # 加载图像
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"无法加载图像 {image_path}: {e}")
            # 如果加载失败，返回下一个样本
            return self.__getitem__((idx + 1) % len(self))
        
        # 应用图像变换
        original_image = self.transform(image)
        
        # 生成 mask
        if self.use_irregular_mask:
            mask_pil = self._generate_irregular_mask(self.image_size, self.image_size)
        else:
            mask_pil = self._generate_rectangle_mask(self.image_size, self.image_size)
        
        # 将 mask 转换为 tensor
        mask = self.mask_transform(mask_pil)
        
        # 二值化 mask
        mask = (mask > 0.5).float()
        
        # 生成遮挡后的图像
        # 在遮挡区域填充白色（值为 1.0，对应归一化后的白色）
        masked_image = original_image * (1 - mask) + mask  # mask 区域变为白色
        
        return original_image, masked_image, mask
    
    def _generate_rectangle_mask(self, height, width):
        """
        生成随机矩形 mask。
        
        参数：
        - height: int, 图像高度
        - width: int, 图像宽度
        
        返回：
        - mask: PIL.Image, 二值 mask（白色区域为待补全区域）
        """
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        # 计算 mask 的尺寸
        mask_ratio = random.uniform(*self.mask_ratio_range)
        mask_area = height * width * mask_ratio
        
        # 随机生成 1-3 个矩形
        num_rectangles = random.randint(1, 3)
        
        for _ in range(num_rectangles):
            # 随机矩形尺寸
            rect_h = int(np.sqrt(mask_area / num_rectangles) * random.uniform(0.5, 2.0))
            rect_w = int(np.sqrt(mask_area / num_rectangles) * random.uniform(0.5, 2.0))
            
            # 确保不超过图像边界
            rect_h = min(rect_h, height - 10)
            rect_w = min(rect_w, width - 10)
            
            # 随机位置
            x1 = random.randint(0, width - rect_w)
            y1 = random.randint(0, height - rect_h)
            x2 = x1 + rect_w
            y2 = y1 + rect_h
            
            # 绘制矩形
            draw.rectangle([x1, y1, x2, y2], fill=255)
        
        return mask
    
    def _generate_irregular_mask(self, height, width):
        """
        生成不规则 mask（随机笔刷线条）。
        
        参数：
        - height: int, 图像高度
        - width: int, 图像宽度
        
        返回：
        - mask: PIL.Image, 二值 mask（白色区域为待补全区域）
        """
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        # 计算 mask 的尺寸
        mask_ratio = random.uniform(*self.mask_ratio_range)
        
        # 绘制多条随机曲线
        num_strokes = random.randint(3, 8)
        
        for _ in range(num_strokes):
            # 随机起点
            x = random.randint(0, width)
            y = random.randint(0, height)
            
            # 随机笔刷宽度
            brush_width = random.randint(15, 50)
            
            # 绘制随机路径
            num_vertices = random.randint(4, 12)
            for _ in range(num_vertices):
                # 随机移动
                angle = random.uniform(0, 2 * np.pi)
                length = random.uniform(20, 100)
                
                x_new = int(x + length * np.cos(angle))
                y_new = int(y + length * np.sin(angle))
                
                # 确保在边界内
                x_new = max(0, min(x_new, width))
                y_new = max(0, min(y_new, height))
                
                # 绘制线段
                draw.line([x, y, x_new, y_new], fill=255, width=brush_width)
                
                x, y = x_new, y_new
        
        return mask


def create_dataloader(
    image_dir,
    ann_file,
    batch_size=4,
    num_workers=4,
    shuffle=True,
    image_size=512,
    mask_ratio_range=(0.1, 0.3),
    use_irregular_mask=True
):
    """
    创建数据加载器的便捷函数。
    
    参数：
    - image_dir: str, 图像文件夹路径
    - ann_file: str, COCO 标注文件路径
    - batch_size: int, 批次大小
    - num_workers: int, 数据加载线程数
    - shuffle: bool, 是否打乱数据
    - image_size: int, 图像尺寸
    - mask_ratio_range: tuple, mask 比例范围
    - use_irregular_mask: bool, 是否使用不规则 mask
    
    返回：
    - dataloader: DataLoader
    """
    dataset = CocoInpaintingDataset(
        image_dir=image_dir,
        ann_file=ann_file,
        image_size=image_size,
        mask_ratio_range=mask_ratio_range,
        use_irregular_mask=use_irregular_mask
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


if __name__ == "__main__":
    """
    测试数据加载器
    
    使用方法：
    python data.py
    """
    import matplotlib.pyplot as plt
    
    # 测试配置（请修改为你的路径）
    IMAGE_DIR = "/path/to/coco/train2017"
    ANN_FILE = "/path/to/coco/annotations/instances_train2017.json"
    
    # 创建数据集
    dataset = CocoInpaintingDataset(
        image_dir=IMAGE_DIR,
        ann_file=ANN_FILE,
        image_size=512,
        use_irregular_mask=True
    )
    
    # 获取一个样本
    original, masked, mask = dataset[0]
    
    print(f"Original image shape: {original.shape}")  # [3, 512, 512]
    print(f"Masked image shape: {masked.shape}")      # [3, 512, 512]
    print(f"Mask shape: {mask.shape}")                # [1, 512, 512]
    print(f"Original image range: [{original.min():.2f}, {original.max():.2f}]")
    print(f"Mask unique values: {mask.unique()}")
    
    # 可视化
    def denormalize(tensor):
        """将 [-1, 1] 的 tensor 转换为 [0, 1]"""
        return tensor * 0.5 + 0.5
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图
    axes[0].imshow(denormalize(original).permute(1, 2, 0).cpu().numpy())
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 遮挡图
    axes[1].imshow(denormalize(masked).permute(1, 2, 0).cpu().numpy())
    axes[1].set_title('Masked Image')
    axes[1].axis('off')
    
    # Mask
    axes[2].imshow(mask.squeeze().cpu().numpy(), cmap='gray')
    axes[2].set_title('Mask (white = inpaint)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('data_sample.png')
    print("样本已保存到 data_sample.png")

