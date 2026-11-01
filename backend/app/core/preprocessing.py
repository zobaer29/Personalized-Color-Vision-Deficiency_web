"""
Image Preprocessing Pipeline for Personalized CVD Recoloring.
Directly aligned with research notebook data pipeline:
- Validates and decodes image input (Base64 / bytes / PIL)
- Corrects EXIF orientation
- Converts to RGB
- Handles large images (safe downscaling if exceeding 2048x2048 while preserving aspect ratio)
- Maps pixel intensities to [0.0, 1.0] float32 tensor
- Pads spatial dimensions to multiple of 8 (required for hierarchical encoder downsampling)
- Condition encoding via spatial one-hot channel broadcasting
"""

import base64
import io
from typing import Tuple, Union
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
import torch.nn.functional as F

from .model import append_condition_channels


MAX_IMAGE_DIMENSION = 2048
MAX_RAW_BYTES = 10 * 1024 * 1024  # 10 MB limit


def decode_base64_image(image_base64: str) -> Tuple[Image.Image, bytes]:
    """
    Decodes a Base64 data URL or raw Base64 string to a validated PIL Image and raw bytes.
    Applies EXIF orientation correction and converts to RGB.
    """
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(image_base64, validate=True)
    except Exception as exc:
        raise ValueError(f"Malformed base64 string: {exc}") from exc

    if len(raw_bytes) > MAX_RAW_BYTES:
        raise ValueError(f"Image payload too large ({len(raw_bytes)} bytes > {MAX_RAW_BYTES} bytes)")

    try:
        pil_image = Image.open(io.BytesIO(raw_bytes))
        pil_image.load()
    except Exception as exc:
        raise UnidentifiedImageError(f"Cannot identify image from payload: {exc}") from exc

    # EXIF orientation correction
    pil_image = ImageOps.exif_transpose(pil_image)

    # Convert to RGB
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Safe dimension downscaling if oversized while preserving aspect ratio
    w, h = pil_image.size
    if max(w, h) > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(w, h)
        new_w = max(8, int(round(w * scale)))
        new_h = max(8, int(round(h * scale)))
        pil_image = pil_image.resize((new_w, new_h), Image.Resampling.BICUBIC)

    return pil_image, raw_bytes


def prepare_input_tensor(
    image: Union[Image.Image, np.ndarray],
    device: torch.device,
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    """
    Converts RGB image to normalized float32 tensor [0.0, 1.0],
    and pads spatial dimensions (H, W) to multiples of 8.

    Returns:
        padded_tensor: (1, 3, H_pad, W_pad)
        original_size: (H, W)
        pad_amounts: (pad_h, pad_w)
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image, dtype=np.float32) / 255.0
    else:
        img_np = image.astype(np.float32) / 255.0

    # (H, W, C) -> (1, C, H, W)
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    h, w = tensor.shape[-2:]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8

    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")

    return tensor, (h, w), (pad_h, pad_w)
