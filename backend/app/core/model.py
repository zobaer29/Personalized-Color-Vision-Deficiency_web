"""
Modified Conditional Restormer Architecture for Multi-Type CVD Recolorization.
Extracted from: a-lightweight-conditional-restormer-for-multi-type.ipynb
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def append_condition_channels(
    image: torch.Tensor,
    condition_id: torch.Tensor,
    number_of_conditions: int = 3,
) -> torch.Tensor:
    """
    Appends spatially broadcasted one-hot condition channels to input image tensor.
    image: (B, 3, H, W)
    condition_id: (B,)
    returns: (B, 3 + number_of_conditions, H, W)
    """
    one_hot = F.one_hot(
        condition_id,
        num_classes=number_of_conditions,
    ).float()

    condition_map = one_hot[:, :, None, None].expand(
        -1,
        -1,
        image.shape[-2],
        image.shape[-1],
    )

    return torch.cat([image, condition_map], dim=1)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        variance = features.var(dim=-1, unbiased=False, keepdim=True)
        return features / torch.sqrt(variance + 1e-5) * self.weight


class LayerNorm2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.normalization = BiasFreeLayerNorm(channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        height, width = features.shape[-2:]
        sequence = features.flatten(2).transpose(1, 2)
        sequence = self.normalization(sequence)
        return sequence.transpose(1, 2).reshape(
            features.shape[0],
            features.shape[1],
            height,
            width,
        )


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention."""
    def __init__(self, channels: int, number_of_heads: int):
        super().__init__()
        if channels % number_of_heads != 0:
            raise ValueError("Channels must be divisible by the number of heads.")

        self.number_of_heads = number_of_heads
        self.temperature = nn.Parameter(torch.ones(number_of_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_depthwise = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            padding=1,
            groups=channels * 3,
            bias=False,
        )
        self.output_projection = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = features.shape
        query, key, value = self.qkv_depthwise(self.qkv(features)).chunk(3, dim=1)
        channels_per_head = channels // self.number_of_heads

        def reshape_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch_size,
                self.number_of_heads,
                channels_per_head,
                height * width,
            )

        query = F.normalize(reshape_heads(query), dim=-1)
        key = F.normalize(reshape_heads(key), dim=-1)
        value = reshape_heads(value)

        attention = (query @ key.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)

        output = attention @ value
        output = output.reshape(batch_size, channels, height, width)
        return self.output_projection(output)


class GatedDepthwiseFeedForward(nn.Module):
    """Gated Depthwise Feed-Forward Network."""
    def __init__(self, channels: int, expansion_factor: float = 2.0):
        super().__init__()
        hidden_channels = int(channels * expansion_factor)
        self.input_projection = nn.Conv2d(
            channels,
            hidden_channels * 2,
            kernel_size=1,
            bias=False,
        )
        self.depthwise = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_channels * 2,
            bias=False,
        )
        self.output_projection = nn.Conv2d(
            hidden_channels,
            channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.depthwise(self.input_projection(features))
        first, second = projected.chunk(2, dim=1)
        return self.output_projection(F.gelu(first) * second)


class RestormerBlock(nn.Module):
    def __init__(self, channels: int, number_of_heads: int):
        super().__init__()
        self.norm1 = LayerNorm2D(channels)
        self.attention = MDTA(channels, number_of_heads)
        self.norm2 = LayerNorm2D(channels)
        self.feed_forward = GatedDepthwiseFeedForward(channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features + self.attention(self.norm1(features))
        features = features + self.feed_forward(self.norm2(features))
        return features


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Channel + Spatial)."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        avg_pool = F.adaptive_avg_pool2d(features, 1)
        max_pool = F.adaptive_max_pool2d(features, 1)
        channel_att = torch.sigmoid(self.channel_mlp(avg_pool) + self.channel_mlp(max_pool))
        features = features * channel_att

        spatial_input = torch.cat(
            [features.mean(dim=1, keepdim=True), features.max(dim=1, keepdim=True).values],
            dim=1,
        )
        spatial_att = torch.sigmoid(self.spatial(spatial_input))
        return features * spatial_att


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block for multi-scale feature enhancement."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1)
        self.fusion = nn.Conv2d(channels * 4, channels, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        first = F.gelu(self.conv1(features))
        second = F.gelu(self.conv2(torch.cat([features, first], dim=1)))
        third = F.gelu(self.conv3(torch.cat([features, first, second], dim=1)))
        fused = self.fusion(torch.cat([features, first, second, third], dim=1))
        return features + fused


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.body(features)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.body(features)


class ModifiedConditionalRestormer(nn.Module):
    """
    Lightweight Conditional Restormer for Multi-Type Colour Vision Deficiency Recolorization.
    """
    def __init__(
        self,
        base_channels: int = 16,
        heads: Tuple[int, int, int] = (1, 2, 4),
        blocks: Tuple[int, int, int] = (1, 1, 2),
        use_cbam: bool = True,
        use_rdb: bool = True,
    ):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.use_cbam = use_cbam
        self.use_rdb = use_rdb

        # Input: 3 RGB + 3 condition channels = 6 channels
        self.patch_embedding = nn.Conv2d(6, c1, kernel_size=3, padding=1, bias=False)

        self.encoder1 = nn.Sequential(*[RestormerBlock(c1, heads[0]) for _ in range(blocks[0])])
        self.down1 = Downsample(c1)

        self.encoder2 = nn.Sequential(*[RestormerBlock(c2, heads[1]) for _ in range(blocks[1])])
        self.down2 = Downsample(c2)

        self.latent = nn.Sequential(*[RestormerBlock(c3, heads[2]) for _ in range(blocks[2])])

        self.rdb = ResidualDenseBlock(c3) if use_rdb else nn.Identity()
        self.cbam = CBAM(c3) if use_cbam else nn.Identity()

        self.up2 = Upsample(c3)
        self.reduce2 = nn.Conv2d(c2 + c2, c2, kernel_size=1, bias=False)
        self.decoder2 = nn.Sequential(*[RestormerBlock(c2, heads[1]) for _ in range(blocks[1])])

        self.up1 = Upsample(c2)
        self.reduce1 = nn.Conv2d(c1 + c1, c1, kernel_size=1, bias=False)
        self.decoder1 = nn.Sequential(*[RestormerBlock(c1, heads[0]) for _ in range(blocks[0])])

        self.output_projection = nn.Conv2d(c1, 3, kernel_size=3, padding=1, bias=False)

    def forward(self, image: torch.Tensor, condition_id: torch.Tensor) -> torch.Tensor:
        conditioned = append_condition_channels(image, condition_id)

        level1 = self.encoder1(self.patch_embedding(conditioned))
        level2 = self.encoder2(self.down1(level1))
        latent = self.latent(self.down2(level2))
        latent = self.cbam(self.rdb(latent))

        decoded2 = self.up2(latent)
        decoded2 = self.reduce2(torch.cat([decoded2, level2], dim=1))
        decoded2 = self.decoder2(decoded2)

        decoded1 = self.up1(decoded2)
        decoded1 = self.reduce1(torch.cat([decoded1, level1], dim=1))
        decoded1 = self.decoder1(decoded1)

        residual = 0.5 * torch.tanh(self.output_projection(decoded1))
        return torch.clamp(image + residual, 0.0, 1.0)


class ResidualCNNBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1

        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.gelu(features + self.body(features))


class PlainConditionalResidualCNN(nn.Module):
    """Model A: Plain Conditional Residual CNN baseline from notebook."""
    def __init__(self, channels: int = 48, number_of_blocks: int = 8):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv2d(6, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualCNNBlock(channels) for _ in range(number_of_blocks)])
        self.output_projection = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, image: torch.Tensor, condition_id: torch.Tensor) -> torch.Tensor:
        conditioned = append_condition_channels(image, condition_id)
        features = self.blocks(self.input_projection(conditioned))
        residual = 0.5 * torch.tanh(self.output_projection(features))
        return torch.clamp(image + residual, 0.0, 1.0)


class DoubleConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        groups = min(8, output_channels)
        while output_channels % groups != 0:
            groups -= 1

        self.body = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.body(features)


class ConditionalResidualUNet(nn.Module):
    """Model B: Conditional Residual U-Net baseline from notebook."""
    def __init__(self, base_channels: int = 32):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.encoder1 = DoubleConv(6, c1)
        self.encoder2 = DoubleConv(c1, c2)
        self.encoder3 = DoubleConv(c2, c3)
        self.bottleneck = DoubleConv(c3, c4)

        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(c3 + c3, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(c2 + c2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(c1 + c1, c1)

        self.output_projection = nn.Conv2d(c1, 3, kernel_size=1)

    def forward(self, image: torch.Tensor, condition_id: torch.Tensor) -> torch.Tensor:
        conditioned = append_condition_channels(image, condition_id)

        encoder1 = self.encoder1(conditioned)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        bottleneck = self.bottleneck(self.pool(encoder3))

        decoder3 = self.decoder3(torch.cat([self.up3(bottleneck), encoder3], dim=1))
        decoder2 = self.decoder2(torch.cat([self.up2(decoder3), encoder2], dim=1))
        decoder1 = self.decoder1(torch.cat([self.up1(decoder2), encoder1], dim=1))

        residual = 0.5 * torch.tanh(self.output_projection(decoder1))
        return torch.clamp(image + residual, 0.0, 1.0)


# Restormer variants from paper ablation study
RESTORMER_VARIANTS = {
    "Base_Restormer_Style": {"use_cbam": False, "use_rdb": False},
    "Restormer_CBAM": {"use_cbam": True, "use_rdb": False},
    "Restormer_RDB": {"use_cbam": False, "use_rdb": True},
    "Modified_Conditional_Restormer": {"use_cbam": True, "use_rdb": True},
}

ALL_MODEL_NAMES = (
    "Plain_Conditional_Residual_CNN",
    "Conditional_Residual_UNet",
    *RESTORMER_VARIANTS.keys(),
)


def create_model(
    model_name: str = "Restormer_RDB",
    base_channels: int = 16,
    heads: Tuple[int, int, int] = (1, 2, 4),
    blocks: Tuple[int, int, int] = (1, 1, 2),
    cnn_channels: int = 48,
    cnn_blocks: int = 8,
    unet_base_channels: int = 32,
) -> nn.Module:
    """
    Model factory matching ipynb ablation and proposed architectures.
    Defaults to 'Restormer_RDB' (highest PSNR 22.36 dB and best LPIPS in paper evaluation).
    """
    if model_name == "Plain_Conditional_Residual_CNN":
        return PlainConditionalResidualCNN(channels=cnn_channels, number_of_blocks=cnn_blocks)

    if model_name == "Conditional_Residual_UNet":
        return ConditionalResidualUNet(base_channels=unet_base_channels)

    if model_name in RESTORMER_VARIANTS:
        flags = RESTORMER_VARIANTS[model_name]
        return ModifiedConditionalRestormer(
            base_channels=base_channels,
            heads=heads,
            blocks=blocks,
            use_cbam=flags["use_cbam"],
            use_rdb=flags["use_rdb"],
        )

    raise ValueError(f"Unknown model name: {model_name}. Supported models: {ALL_MODEL_NAMES}")


def count_trainable_parameters(model: nn.Module) -> int:
    """Calculates total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


