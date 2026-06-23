"""
Util functions for network construction.
"""
import torch
import torch.nn.functional as F
import json


def coords_grid(b, h, w, device):
    coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(b, 1, 1, 1)


def load_args_from_json(json_path):
    """Load parameters from a JSON configuration file as an attribute-access object."""
    with open(json_path, 'r') as f:
        config_dict = json.load(f)

    class Args: pass
    args = Args()
    args.__dict__.update(config_dict)
    return args


def flow_warp(input_img, flow):
    """Warp ``input_img`` into the target frame coordinate space.

    Args:
        input_img: [B, C, H, W]
        flow:      [B, 2, H, W], optical flow from input_img to the target frame
    Returns:
        [B, C, H, W] warped image
    """
    B, C, H, W = input_img.shape
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=input_img.device),
        torch.linspace(-1, 1, W, device=input_img.device),
        indexing='ij'
    )
    base_grid = torch.stack((x, y), dim=-1).unsqueeze(0).expand(B, H, W, 2)  # [B, H, W, 2]

    norm_flow = torch.zeros_like(base_grid)
    norm_flow[..., 0] = flow[:, 0, :, :] * (2.0 / (W - 1))
    norm_flow[..., 1] = flow[:, 1, :, :] * (2.0 / (H - 1))

    sample_grid = base_grid + norm_flow
    warped = F.grid_sample(input_img, sample_grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped