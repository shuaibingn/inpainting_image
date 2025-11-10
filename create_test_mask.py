"""
创建测试 mask 的辅助脚本

用于快速生成各种形状的 mask，用于测试推理效果。

使用方法：
python create_test_mask.py --type rectangle --output test_mask.png
python create_test_mask.py --type circle --output test_mask.png
python create_test_mask.py --type irregular --output test_mask.png
"""

import argparse
import numpy as np
from PIL import Image, ImageDraw


def create_rectangle_mask(width, height, num_rects=3):
    """创建矩形 mask"""
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    for _ in range(num_rects):
        x1 = np.random.randint(0, width // 2)
        y1 = np.random.randint(0, height // 2)
        x2 = np.random.randint(width // 2, width)
        y2 = np.random.randint(height // 2, height)
        draw.rectangle([x1, y1, x2, y2], fill=255)
    
    return mask


def create_circle_mask(width, height, num_circles=5):
    """创建圆形 mask"""
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    for _ in range(num_circles):
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        radius = np.random.randint(30, 100)
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=255
        )
    
    return mask


def create_irregular_mask(width, height, num_strokes=8):
    """创建不规则 mask（笔刷线条）"""
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    for _ in range(num_strokes):
        # 随机起点
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        
        # 随机笔刷宽度
        brush_width = np.random.randint(20, 60)
        
        # 绘制随机路径
        num_vertices = np.random.randint(5, 15)
        for _ in range(num_vertices):
            angle = np.random.uniform(0, 2 * np.pi)
            length = np.random.uniform(30, 120)
            
            x_new = int(x + length * np.cos(angle))
            y_new = int(y + length * np.sin(angle))
            
            # 确保在边界内
            x_new = max(0, min(x_new, width))
            y_new = max(0, min(y_new, height))
            
            # 绘制线段
            draw.line([x, y, x_new, y_new], fill=255, width=brush_width)
            
            x, y = x_new, y_new
    
    return mask


def create_center_mask(width, height, mask_size=0.4):
    """创建中心矩形 mask"""
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # 计算中心矩形
    mask_w = int(width * mask_size)
    mask_h = int(height * mask_size)
    x1 = (width - mask_w) // 2
    y1 = (height - mask_h) // 2
    x2 = x1 + mask_w
    y2 = y1 + mask_h
    
    draw.rectangle([x1, y1, x2, y2], fill=255)
    
    return mask


def create_text_mask(width, height, text="TEST"):
    """创建文字形状的 mask"""
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # 使用大字体（需要 PIL 的默认字体）
    from PIL import ImageFont
    try:
        # 尝试使用系统字体
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 150)
    except:
        # 如果失败，使用默认字体
        font = ImageFont.load_default()
    
    # 获取文字边界
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # 居中绘制
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    draw.text((x, y), text, fill=255, font=font)
    
    return mask


def main():
    parser = argparse.ArgumentParser(description="创建测试 mask")
    parser.add_argument(
        "--type",
        type=str,
        default="irregular",
        choices=["rectangle", "circle", "irregular", "center", "text"],
        help="mask 类型"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="mask 宽度"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="mask 高度"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_mask.png",
        help="输出文件路径"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="MASK",
        help="文字 mask 的文本内容（仅在 type=text 时使用）"
    )
    
    args = parser.parse_args()
    
    print(f"创建 {args.type} 类型的 mask...")
    print(f"尺寸: {args.width}x{args.height}")
    
    # 创建 mask
    if args.type == "rectangle":
        mask = create_rectangle_mask(args.width, args.height)
    elif args.type == "circle":
        mask = create_circle_mask(args.width, args.height)
    elif args.type == "irregular":
        mask = create_irregular_mask(args.width, args.height)
    elif args.type == "center":
        mask = create_center_mask(args.width, args.height)
    elif args.type == "text":
        mask = create_text_mask(args.width, args.height, args.text)
    
    # 保存
    mask.save(args.output)
    print(f"✓ Mask 已保存到: {args.output}")
    
    # 统计白色像素比例
    mask_array = np.array(mask)
    white_ratio = (mask_array > 128).sum() / mask_array.size
    print(f"白色区域占比: {white_ratio * 100:.1f}%")


if __name__ == "__main__":
    main()

