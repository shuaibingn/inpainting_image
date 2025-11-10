"""
PyTorch Lightning DataModule

封装数据加载逻辑到 LightningDataModule 中，
保持原有的数据处理逻辑不变。

包含：
- 训练数据加载
- 验证数据加载（可选）
- 数据预处理
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from data import CocoInpaintingDataset


class ControlNetInpaintingDataModule(pl.LightningDataModule):
    """
    ControlNet Inpainting 的 Lightning DataModule。
    
    封装数据集加载、数据增强、DataLoader 创建等逻辑。
    """
    
    def __init__(
        self,
        image_dir,
        ann_file,
        batch_size=4,
        num_workers=4,
        image_size=512,
        mask_ratio_range=(0.1, 0.3),
        use_irregular_mask=True,
        val_split=0.05,  # 验证集比例
        pin_memory=True,
    ):
        """
        初始化 DataModule。
        
        参数：
        - image_dir: str - COCO 图像目录
        - ann_file: str - COCO 标注文件
        - batch_size: int - 批次大小
        - num_workers: int - 数据加载线程数
        - image_size: int - 图像尺寸
        - mask_ratio_range: tuple - mask 占比范围
        - use_irregular_mask: bool - 是否使用不规则 mask
        - val_split: float - 验证集比例（0-1）
        - pin_memory: bool - 是否使用 pin_memory
        """
        super().__init__()
        
        # 保存参数
        self.image_dir = image_dir
        self.ann_file = ann_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.mask_ratio_range = mask_ratio_range
        self.use_irregular_mask = use_irregular_mask
        self.val_split = val_split
        self.pin_memory = pin_memory
        
        # 数据集（在 setup 中初始化）
        self.train_dataset = None
        self.val_dataset = None
    
    def prepare_data(self):
        """
        准备数据（下载、预处理等）。
        
        只在主进程中调用一次。
        在分布式训练中，这个方法只在 rank 0 进程中执行。
        """
        # COCO 数据集通常需要手动下载
        # 这里可以添加自动下载逻辑，但通常数据集已经存在
        pass
    
    def setup(self, stage=None):
        """
        设置数据集。
        
        在每个进程中调用。
        
        参数：
        - stage: str - 'fit', 'validate', 'test', 或 'predict'
        """
        # 创建完整数据集
        full_dataset = CocoInpaintingDataset(
            image_dir=self.image_dir,
            ann_file=self.ann_file,
            image_size=self.image_size,
            mask_ratio_range=self.mask_ratio_range,
            use_irregular_mask=self.use_irregular_mask,
        )
        
        # 分割训练集和验证集
        if self.val_split > 0:
            val_size = int(len(full_dataset) * self.val_split)
            train_size = len(full_dataset) - val_size
            
            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
            )
            
            print(f"数据集划分: 训练集 {train_size} 张，验证集 {val_size} 张")
        else:
            self.train_dataset = full_dataset
            self.val_dataset = None
            
            print(f"数据集大小: 训练集 {len(full_dataset)} 张")
    
    def train_dataloader(self):
        """
        创建训练数据加载器。
        
        返回：
        - DataLoader: 训练数据加载器
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,  # 丢弃最后不完整的 batch
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def val_dataloader(self):
        """
        创建验证数据加载器。
        
        返回：
        - DataLoader: 验证数据加载器（如果有）
        """
        if self.val_dataset is None:
            return None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def teardown(self, stage=None):
        """
        清理资源。
        
        参数：
        - stage: str - 训练阶段
        """
        # 清理操作（如果需要）
        pass


if __name__ == "__main__":
    """
    测试 DataModule
    """
    print("=" * 60)
    print("测试 ControlNet DataModule")
    print("=" * 60)
    
    # 测试配置（请修改为你的路径）
    IMAGE_DIR = "/path/to/coco/train2017"
    ANN_FILE = "/path/to/coco/annotations/instances_train2017.json"
    
    # 创建 DataModule
    datamodule = ControlNetInpaintingDataModule(
        image_dir=IMAGE_DIR,
        ann_file=ANN_FILE,
        batch_size=4,
        num_workers=0,  # 测试时使用 0
        val_split=0.1,
    )
    
    # 设置数据集
    print("\n设置数据集...")
    datamodule.setup()
    
    # 获取数据加载器
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    print(f"\n训练批次数: {len(train_loader)}")
    if val_loader is not None:
        print(f"验证批次数: {len(val_loader)}")
    
    # 获取一个 batch
    print("\n获取一个训练 batch...")
    batch = next(iter(train_loader))
    original, masked, mask = batch
    
    print(f"Original image shape: {original.shape}")
    print(f"Masked image shape: {masked.shape}")
    print(f"Mask shape: {mask.shape}")
    
    print("\n✓ DataModule 测试完成！")
    print("=" * 60)

