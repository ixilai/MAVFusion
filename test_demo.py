# -*- coding: utf-8 -*-
# MAVFusion: demo inference entry point
# Authors: Xilai Li, Weijun Jiang, Xiaosong Li, Yang Liu, Hongbin Wang, Tao Ye, Huafeng Li, Haishu Tan (ECCV 2026)
# Engineering scaffolding adapted from UniVF (Zixiang Zhao et al., NeurIPS 2025 Spotlight).

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import glob
import logging
import sys
import warnings

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.base_two_modal_dataset import DatasetMode
from src.dataset import get_multi_frame_dataset
from src.model.net import VideoFusion
from src.util.io import pred_2_8bit, save_image
from src.util.logging_util import setup_logging
from src.util.video_util import generate_video_from_image_paths

warnings.filterwarnings("ignore")


DATASET_TO_YAML = {
    "VTMOT":      "config/dataset/IVF/VTMOT/vtmot_5-frame.yaml",
    "VTMOT-demo": "config/dataset/demo/vtmot_demo_5-frame.yaml",
}

IVF_CONFIG = "config/train/ivf-train.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="MAVFusion demo inference")
    parser.add_argument("--exp_path", type=str, default="MAVFusion-IVF",
                        help="Experiment directory")
    parser.add_argument("--dataset_name", type=str, default="VTMOT-demo",
                        choices=sorted(DATASET_TO_YAML.keys()),
                        help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size during testing")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run testing on (e.g., 'cuda', 'cuda:0', 'cpu')")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--fps", type=int, default=24, help="Frame rate for generated videos")
    parser.add_argument("--bitrate", type=int, default=50000, help="Bitrate for generated videos")
    return parser.parse_args()


def main():
    args = parse_args()

    vis_dir = os.path.join("output_demo", args.dataset_name, "fused_result")
    setup_logging(
        os.path.dirname(vis_dir),
        file_log_level="INFO",
        console_log_level=args.log_level.upper(),
    )
    logging.info(f"Testing script started with args: {args}")

    device = torch.device(args.device)
    logging.info(f"Using device: {device}")

    args.config = IVF_CONFIG
    args.data_cfg = DATASET_TO_YAML[args.dataset_name]

    cfg = OmegaConf.load(args.config)

    model_path = os.path.join("output", args.exp_path, "checkpoint", "latest", "model.pth")
    if not os.path.exists(model_path):
        logging.error(f"Model checkpoint not found: {model_path}")
        sys.exit(1)

    logging.info("Initializing model...")
    model = VideoFusion(model_config=cfg).to(device).eval()
    model.load_state_dict(torch.load(model_path, map_location=device))
    logging.info(f"Model weights successfully loaded from {model_path}")

    logging.info(f"Loading test dataset: {args.data_cfg}")
    data_cfg = OmegaConf.load(args.data_cfg)
    dataset = get_multi_frame_dataset(data_cfg, base_data_dir="data", mode=DatasetMode.TEST)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    logging.info(f"Dataset '{args.dataset_name}' loaded with {len(dataset)} samples.")

    os.makedirs(vis_dir, exist_ok=True)

    logging.info("Starting inference loop...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Testing {args.dataset_name}"):
            ir_img = batch["ir"].to(device)
            rgb_img = batch["rgb"].to(device)

            fusion_pred, _ = model(ir_img, rgb_img)
            if torch.isnan(fusion_pred).any():
                logging.warning("fusion_pred contains NaN values.")

            vis_8bit = pred_2_8bit(fusion_pred, ir_img, rgb_img)
            middle_idx = ir_img.shape[1] // 2
            ir_path = batch["data_path_ls_dict"]["ir"][middle_idx][0]
            seq_name = os.path.dirname(ir_path).split(os.sep)[0]
            frame_stem = os.path.splitext(os.path.basename(ir_path))[0]
            file_name = f"{seq_name}_{frame_stem}.png"
            save_image(vis_8bit, os.path.join(vis_dir, file_name))

    logging.info("Merging images into video...")
    filenames = sorted(glob.glob(os.path.join(vis_dir, "*.png")))
    logging.info(f"Found {len(filenames)} images")
    output_video_path = os.path.join(os.path.dirname(vis_dir), "fused_result.mp4")
    generate_video_from_image_paths(
        image_paths=filenames,
        output_path=output_video_path,
        fps=args.fps,
        bitrate=args.bitrate,
        verbose=True,
    )


if __name__ == "__main__":
    main()