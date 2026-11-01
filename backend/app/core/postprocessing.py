"""
Image Postprocessing Pipeline for Personalized CVD Recoloring.
Directly handles:
- Cropping padded dimensions back to exact original (H, W)
- Clamping tensor values to valid [0.0, 1.0] range
- Converting tensor to uint8 numpy RGB array [0, 255]
- Encoding numpy array to Base64 data URL with WebP, PNG, or JPEG support
"""

import base64
import io
from typing import Tuple, Literal
import numpy as np
from PIL import Image
import torch


SUPPORTED_FORMATS = {"webp", "jpeg", "jpg", "png"}


def unpad_and_clamp_tensor(
    output_tensor: torch.Tensor,
    original_size: Tuple[int, int],
) -> np.ndarray:
    """
    Crops output tensor back to original (H, W), clamps to [0.0, 1.0],
    and converts to uint8 (H, W, 3) numpy array.
    """
    orig_h, orig_w = original_size
    cropped = output_tensor[0, :, :orig_h, :orig_w]
    clamped = cropped.detach().cpu().clamp(0.0, 1.0)
    # (3, H, W) -> (H, W, 3)
    np_img = (clamped.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return np_img


def encode_base64_image(
    image_np: np.ndarray,
    format_type: str = "webp",
    quality: int = 90,
) -> str:
    """
    Encodes an RGB numpy array to a Base64 data URL string.
    Supported formats: 'webp', 'jpeg', 'png'.
    """
    fmt = format_type.strip().lower()
    if fmt == "jpg":
        fmt = "jpeg"

    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format '{format_type}'. Supported: {sorted(SUPPORTED_FORMATS)}")

    pil_img = Image.fromarray(image_np)
    buffer = io.BytesIO()

    if fmt == "webp":
        pil_img.save(buffer, format="WEBP", quality=quality, method=4)
        prefix = "data:image/webp;base64,"
    elif fmt == "jpeg":
        pil_img.save(buffer, format="JPEG", quality=quality)
        prefix = "data:image/jpeg;base64,"
    elif fmt == "png":
        pil_img.save(buffer, format="PNG", optimize=True)
        prefix = "data:image/png;base64,"

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return prefix + encoded
