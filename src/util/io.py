# MAVFusion: I/O utilities for inference visualization
# Authors: Xilai Li, Weijun Jiang, Xiaosong Li, Yang Liu, Hongbin Wang, Tao Ye, Huafeng Li, Haishu Tan (ECCV 2026)

import csv
import torch
import numpy as np
import einops
import cv2
from PIL import Image


def read_csv(filename, delimiter=","):
    with open(filename, "r", newline="") as f:
        csv_reader = csv.reader(f, delimiter=delimiter)
        header = next(csv_reader)
        content = [row for row in csv_reader if row]
    return header, content


def pred_2_8bit(
    pred: torch.Tensor,
    src1: torch.Tensor,
    src2: torch.Tensor,
    inherit_vi_chroma: bool = True,
) -> np.ndarray:
    """
    Convert the middle fused frame to a uint8 image for saving.

    Args:
        pred: fused output, shape ``[B, 3, 3, H, W]`` (3 fused middle frames, RGB).
        src1: IR source, shape ``[B, 5, 3, H, W]``.
        src2: VI source, shape ``[B, 5, 3, H, W]``.
        inherit_vi_chroma: when True (default), the fused RGB is converted to
            YCbCr, the luma (Y) channel is kept, and the chroma (Cb/Cr) channels
            are replaced with the VI frame's Cb/Cr before converting back to
            RGB. The network's 3-channel output is only trustworthy for
            luminance/structure; inheriting VI's chroma gives much better color
            fidelity than letting the network hallucinate color.

    Returns:
        uint8 RGB image, shape ``[H, W, 3]``.
    """
    fused = pred[:, 1, :, :, :].detach().cpu().clone().numpy().squeeze()  # [3, H, W]
    assert fused.ndim == 3, "pred should be a 3D tensor"

    fused = einops.rearrange(fused, "c h w -> h w c")
    fused_uint8 = np.clip(fused * 255.0, 0, 255).astype(np.uint8)  # [H, W, 3] RGB

    if not inherit_vi_chroma:
        return fused_uint8

    # Middle VI frame (rgb, [0, 1]) — used only for its Cb/Cr channels.
    vi = src2[:, 2, :, :, :].detach().cpu().clone().numpy().squeeze()
    vi = einops.rearrange(vi, "c h w -> h w c")
    vi_uint8 = np.clip(vi * 255.0, 0, 255).astype(np.uint8)

    # BT.601 full-range conversion: cv2 returns YCrCb (note the channel order).
    fused_ycrcb = cv2.cvtColor(fused_uint8, cv2.COLOR_RGB2YCrCb)
    vi_ycrcb = cv2.cvtColor(vi_uint8, cv2.COLOR_RGB2YCrCb)

    # Replace chroma with VI's; keep fused luminance.
    fused_ycrcb[..., 1] = vi_ycrcb[..., 1]  # Cr
    fused_ycrcb[..., 2] = vi_ycrcb[..., 2]  # Cb

    return cv2.cvtColor(fused_ycrcb, cv2.COLOR_YCrCb2RGB)


def save_image(image: np.ndarray, filename: str):
    if image.ndim == 2:  # Grayscale
        image = np.expand_dims(image, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:  # Single channel
        image = np.squeeze(image, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 3:  # RGB
        pass
    else:
        raise ValueError("Image must be either grayscale or RGB format.")

    im = Image.fromarray(image)
    im.save(filename)
