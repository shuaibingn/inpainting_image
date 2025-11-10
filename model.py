"""
ControlNet 模型实现 - 用于图像补全（Inpainting）

基于 lllyasviel/ControlNet 的原始实现，包含：
- UNet 主干（从 Stable Diffusion v1.5 加载）
- ControlNet 分支（复制 UNet 结构）
- Zero Convolution（零初始化卷积层）
- 特征注入机制

模型输入：
- x_t: [B, 3, H, W] - 加噪后的图像
- t: [B] - 扩散时间步
- control_input: [B, 4, H, W] - 控制输入 [masked_image, mask]

模型输出：
- pred_noise: [B, 3, H, W] - 预测的噪声
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor
import copy


class ZeroConv(nn.Module):
    """
    零初始化卷积层。
    
    用于 ControlNet 特征注入，初始状态下不影响主干网络。
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super(ZeroConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            padding=padding
        )
        # 零初始化
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
    
    def forward(self, x):
        return self.conv(x)


class ControlNetConditioningEmbedding(nn.Module):
    """
    ControlNet 条件编码模块。
    
    将 4 通道输入 [masked_image, mask] 编码为特征。
    """
    def __init__(
        self,
        conditioning_embedding_channels=320,
        conditioning_channels=4,
        block_out_channels=(16, 32, 96, 256),
    ):
        super().__init__()
        
        self.conv_in = nn.Conv2d(
            conditioning_channels, 
            block_out_channels[0], 
            kernel_size=3, 
            padding=1
        )
        
        self.blocks = nn.ModuleList([])
        
        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2)
            )
        
        self.conv_out = ZeroConv(
            block_out_channels[-1],
            conditioning_embedding_channels,
            kernel_size=3,
            padding=1
        )
    
    def forward(self, conditioning):
        """
        参数：
        - conditioning: [B, 4, H, W] - 控制输入
        
        返回：
        - embedding: [B, 320, H, W] - 编码后的特征
        """
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)
        
        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)
        
        embedding = self.conv_out(embedding)
        
        return embedding


class ControlNetModel(nn.Module):
    """
    ControlNet 模型。
    
    包含完整的 ControlNet 结构，与 UNet 相同的架构，
    用于从控制输入中提取特征并注入到主干 UNet。
    """
    def __init__(self, unet):
        """
        参数：
        - unet: UNet2DConditionModel - 预训练的 UNet 模型
        """
        super().__init__()
        
        # 复制 UNet 的结构和权重来初始化 ControlNet
        self.controlnet = copy.deepcopy(unet)
        
        # 条件编码模块（处理 4 通道输入）
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=320,
            conditioning_channels=4,  # [masked_image (3) + mask (1)]
            block_out_channels=(16, 32, 96, 256),
        )
        
        # 获取 UNet 的 down_blocks 和 mid_block 配置
        # 为每个输出创建 zero convolution
        self.controlnet_down_blocks = nn.ModuleList([])
        
        # 下采样块的 zero convolution
        for i, down_block in enumerate(self.controlnet.down_blocks):
            # 获取输出通道数
            if hasattr(down_block, 'resnets'):
                out_channels = down_block.resnets[-1].out_channels
            else:
                out_channels = 320  # 默认值
            
            zero_conv = ZeroConv(out_channels, out_channels, kernel_size=1)
            self.controlnet_down_blocks.append(zero_conv)
        
        # 中间块的 zero convolution
        if hasattr(self.controlnet.mid_block, 'resnets'):
            mid_block_channels = self.controlnet.mid_block.resnets[-1].out_channels
        else:
            mid_block_channels = 1280  # 默认值
        
        self.controlnet_mid_block = ZeroConv(
            mid_block_channels, 
            mid_block_channels, 
            kernel_size=1
        )
    
    def forward(self, x, timestep, encoder_hidden_states, conditioning):
        """
        前向传播。
        
        参数：
        - x: [B, 4, H, W] - 加噪后的图像（latent）
        - timestep: [B] - 时间步
        - encoder_hidden_states: [B, seq_len, dim] - 文本编码（可以为 None）
        - conditioning: [B, 4, H, W] - 控制输入 [masked_image, mask]
        
        返回：
        - down_block_res_samples: list of tensors - 下采样块的特征
        - mid_block_res_sample: tensor - 中间块的特征
        """
        # 编码条件输入
        cond_embedding = self.controlnet_cond_embedding(conditioning)
        
        # 将条件嵌入加到输入上
        x = x + cond_embedding
        
        # 时间步嵌入
        t_emb = self.controlnet.time_proj(timestep)
        emb = self.controlnet.time_embedding(t_emb)
        
        # 初始卷积
        h = self.controlnet.conv_in(x)
        
        # 下采样
        down_block_res_samples = []
        
        for i, downsample_block in enumerate(self.controlnet.down_blocks):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                h, res_samples = downsample_block(
                    hidden_states=h,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                h, res_samples = downsample_block(
                    hidden_states=h,
                    temb=emb,
                )
            
            # 对最后一个 residual 应用 zero convolution
            if len(res_samples) > 0:
                res_sample = res_samples[-1]
                res_sample = self.controlnet_down_blocks[i](res_sample)
                down_block_res_samples.append(res_sample)
        
        # 中间块
        h = self.controlnet.mid_block(
            h,
            emb,
            encoder_hidden_states=encoder_hidden_states,
        )
        
        # 应用 zero convolution 到中间块输出
        mid_block_res_sample = self.controlnet_mid_block(h)
        
        return down_block_res_samples, mid_block_res_sample


class ControlNetInpaintingModel(nn.Module):
    """
    完整的 ControlNet Inpainting 模型。
    
    包含：
    - UNet 主干（冻结权重）
    - ControlNet 分支（可训练）
    - 特征注入机制
    """
    def __init__(
        self, 
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        freeze_unet=True
    ):
        """
        初始化模型。
        
        参数：
        - pretrained_model_name_or_path: str - 预训练模型路径
        - freeze_unet: bool - 是否冻结 UNet 权重（默认 True）
        """
        super().__init__()
        
        print(f"正在加载预训练模型：{pretrained_model_name_or_path}")
        
        # 加载预训练的 UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet"
        )
        
        # 创建 ControlNet
        self.controlnet = ControlNetModel(self.unet)
        
        # 冻结 UNet 权重
        if freeze_unet:
            print("冻结 UNet 权重")
            for param in self.unet.parameters():
                param.requires_grad = False
        
        print("模型初始化完成")
    
    def forward(self, x_t, timestep, control_input, encoder_hidden_states=None):
        """
        前向传播。
        
        参数：
        - x_t: [B, 4, H, W] - 加噪后的图像（latent space）
        - timestep: [B] or int - 扩散时间步
        - control_input: [B, 4, H, W] - 控制输入 [masked_image, mask]
        - encoder_hidden_states: [B, seq_len, dim] - 文本编码（可选）
        
        返回：
        - noise_pred: [B, 4, H, W] - 预测的噪声
        """
        # 如果 timestep 是标量，扩展为 batch
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=x_t.device)
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).repeat(x_t.shape[0])
        
        # 获取 ControlNet 特征
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            x=x_t,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            conditioning=control_input
        )
        
        # UNet 前向传播，注入 ControlNet 特征
        noise_pred = self.unet(
            x_t,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample
        
        return noise_pred


def create_model(
    pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
    freeze_unet=True
):
    """
    创建 ControlNet Inpainting 模型的便捷函数。
    
    参数：
    - pretrained_model_name_or_path: str - 预训练模型路径
    - freeze_unet: bool - 是否冻结 UNet 权重
    
    返回：
    - model: ControlNetInpaintingModel
    """
    model = ControlNetInpaintingModel(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        freeze_unet=freeze_unet
    )
    return model


if __name__ == "__main__":
    """
    测试模型结构
    
    使用方法：
    python model.py
    """
    import torch
    
    print("=" * 60)
    print("测试 ControlNet Inpainting 模型")
    print("=" * 60)
    
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 注意：首次运行会下载预训练模型，需要一些时间
    model = create_model(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        freeze_unet=True
    )
    model = model.to(device)
    model.eval()
    
    # 创建测试输入
    batch_size = 2
    
    # Stable Diffusion 在 latent space 中操作（64x64），不是 512x512
    # latent 是 4 通道的
    x_t = torch.randn(batch_size, 4, 64, 64).to(device)
    
    # 时间步
    timestep = torch.randint(0, 1000, (batch_size,)).to(device)
    
    # 控制输入（在 latent space 中也是 4 通道）
    # 注意：实际使用中，需要用 VAE 将图像编码到 latent space
    control_input = torch.randn(batch_size, 4, 64, 64).to(device)
    
    print(f"\n输入形状：")
    print(f"  x_t: {x_t.shape}")
    print(f"  timestep: {timestep.shape}")
    print(f"  control_input: {control_input.shape}")
    
    # 前向传播
    with torch.no_grad():
        noise_pred = model(x_t, timestep, control_input)
    
    print(f"\n输出形状：")
    print(f"  noise_pred: {noise_pred.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数统计：")
    print(f"  总参数量: {total_params / 1e6:.2f}M")
    print(f"  可训练参数: {trainable_params / 1e6:.2f}M")
    print(f"  冻结参数: {(total_params - trainable_params) / 1e6:.2f}M")
    
    print("\n" + "=" * 60)
    print("模型测试完成！")
    print("=" * 60)

