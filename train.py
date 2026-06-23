# -*- coding: utf-8 -*-
# MAVFusion: training entry point
# Authors: Xilai Li, Weijun Jiang, Xiaosong Li, Yang Liu, Hongbin Wang, Tao Ye, Huafeng Li, Haishu Tan (ECCV 2026)
# Engineering scaffolding adapted from UniVF (Zixiang Zhao et al., NeurIPS 2025 Spotlight).

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] ="expandable_segments:True"
import argparse
import logging
import os
import shutil
from datetime import datetime, timedelta
from typing import List
import pytz  # timezone
import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from accelerate import Accelerator  # Multi-GPU training and mixed precision
from src.dataset import get_multi_frame_dataset, BaseTwoModalDataset, DatasetMode
from src.util.config_util import recursive_load_config
from src.util.logging_util import (
    config_logging,
    init_wandb,
    load_wandb_job_id,
    log_slurm_job_id,
    save_wandb_job_id,
    tb_logger,
    create_code_snapshot,
)
from src.model.net import VideoFusion
from pathlib import Path
from tools.prepare_dataset import run as prepare_dataset_run
import warnings
warnings.filterwarnings("ignore")


def _resolve_from_raw_data_dir(args):
    """If --raw_data_dir is set, run dataset prep and override base_data_dir.

    Returns (base_data_dir, dataset_yaml_path_or_None).
    """
    if not getattr(args, "raw_data_dir", None):
        return args.base_data_dir, None

    raw_path = Path(args.raw_data_dir).expanduser().resolve()
    if not raw_path.is_dir():
        raise NotADirectoryError(f"--raw_data_dir is not a directory: {raw_path}")

    modality_dirs = (
        tuple(args.raw_modality_dirs)
        if getattr(args, "raw_modality_dirs", None)
        else ("infrared", "visible")
    )

    outputs = prepare_dataset_run(
        source=str(raw_path),
        task_name="IVF",
        dataset_name=getattr(args, "raw_dataset_name", None),
        modality_dirs=modality_dirs,
        train_ratio=getattr(args, "raw_train_ratio", 0.8),
        split_seed=getattr(args, "raw_split_seed", 2025),
        file_ext=getattr(args, "raw_file_ext", ".jpg"),
        force=getattr(args, "force_prep", False),
    )
    base_data_dir = str(raw_path.parent) + "/"
    return base_data_dir, str(outputs["yaml"])


if "__main__" == __name__:

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(description="Train MAVFusion (IVIF)")
    parser.add_argument(
        "--resume_run",
        action="store",
        default=None,
        help="Path of checkpoint to be resumed. If given, will ignore --config, and checkpoint in the config",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="directory to save checkpoints"
    )
    parser.add_argument(
        "--mixed_precision", type=str, default="no", choices=["no", "bf16", "fp16"]
    )
    parser.add_argument("--no_wandb", action="store_true", help="run without wandb")
    parser.add_argument(
        "--base_data_dir", type=str, default="./data2", help="directory of training data"
    )
    parser.add_argument(
        "--add_datetime_prefix",
        action="store_false",
        help="Add datetime to the output folder name",
    )
    parser.add_argument(
        "--split_batch",
        action="store_true",
        help="Accelerator split batch",
    )
    # ---- one-shot data prep shortcuts ----
    parser.add_argument(
        "--raw_data_dir", type=str, default=None,
        help="Path to a directory of paired multi-modal video frames; the prep "
             "tool will auto-generate split.json / CSVs / dataset YAML on the fly.",
    )
    parser.add_argument(
        "--raw_modality_dirs", nargs=2, default=None, metavar=("MOD_A", "MOD_B"),
        help="Subdirectory names for the two modalities under each sequence (default: infrared visible).",
    )
    parser.add_argument(
        "--raw_train_ratio", type=float, default=0.8,
        help="Fraction of sequences to use for training (when --raw_data_dir is set).",
    )
    parser.add_argument(
        "--raw_split_seed", type=int, default=2025,
        help="Random seed for the auto-generated train/test split.",
    )
    parser.add_argument(
        "--raw_dataset_name", type=str, default=None,
        help="Dataset name for generated files (default: basename of --raw_data_dir).",
    )
    parser.add_argument(
        "--raw_file_ext", type=str, default=".jpg",
        help="Image extension to include when auto-discovering frames.",
    )
    parser.add_argument(
        "--force_prep", action="store_true",
        help="Overwrite existing split.json / CSVs when --raw_data_dir is set.",
    )

    # args = parser.parse_args()
    args, unknown_args = parser.parse_known_args()
    resume_run = args.resume_run
    output_dir = args.output_dir

    # Resolve --raw_data_dir before any other initialization so that the
    # generated YAML is available when the main config is loaded below.
    base_data_dir, raw_yaml = _resolve_from_raw_data_dir(args)
    if base_data_dir is None:
        base_data_dir = (
            args.base_data_dir
            if args.base_data_dir is not None
            else os.environ["BASE_DATA_DIR"]
        )

    # -------------------- Accelerator --------------------
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision, split_batches=args.split_batch
    )

    # -------------------- Initialization --------------------
    t_start = datetime.now(pytz.timezone("Europe/Zurich"))
    # Resume previous run
    if resume_run is not None:
        print(f"Resume run: {resume_run}")
        out_dir_run = os.path.dirname(os.path.dirname(resume_run))
        job_name = os.path.basename(out_dir_run)
        # Resume config file
        cfg = OmegaConf.load(os.path.join(out_dir_run, "config.yaml"))
    else:
        # Load config
        config_path = "config/train/ivf-train.yaml"
        cfg = recursive_load_config(config_path, unknown_args)

        # If --raw_data_dir was used, point the train dataset config at the
        # auto-generated YAML produced by tools/prepare_dataset.py.
        if raw_yaml is not None:
            cfg.dataset_cfg.train = raw_yaml

        # Output folder name
        task_tag = "IVF"
        if args.add_datetime_prefix:
            job_name = (
                f"{t_start.strftime('%y_%m_%d-%H_%M_%S')}"
                f"-{task_tag}"
                f"-crop{cfg.augmentation.random_crop_hw[0]}"
                f"-bs{cfg.dataloader.effective_batch_size}_{cfg.dataloader.max_train_batch_size}"
                f"-coef{'_'.join(map(str, cfg.loss.kwargs.coef))}"
                f"-lr{cfg.lr:.0e}"
            )
        else:
            job_name = (
                f"-{task_tag}"
                f"-crop{cfg.augmentation.random_crop_hw[0]}"
                f"-bs{cfg.dataloader.effective_batch_size}_{cfg.dataloader.max_train_batch_size}"
                f"-coef{'_'.join(map(str, cfg.loss.kwargs.coef))}"
                f"-lr{cfg.lr:.0e}"
            )

        if "no" != args.mixed_precision:
            job_name += f"_{args.mixed_precision}"

        # Output directory
        if output_dir is not None:
            out_dir_run = os.path.join(output_dir, job_name)
        else:
            out_dir_run = os.path.join("./output", job_name)
        if accelerator.is_main_process:
            os.makedirs(out_dir_run, exist_ok=False)

    # Other directories
    out_dir_ckpt = os.path.join(out_dir_run, "checkpoint")
    out_dir_tb = os.path.join(out_dir_run, "tensorboard")
    out_dir_eval = os.path.join(out_dir_run, "evaluation")
    out_dir_vis = os.path.join(out_dir_run, "visualization")
    if accelerator.is_main_process:
        if not os.path.exists(out_dir_ckpt):
            os.makedirs(out_dir_ckpt)
        if not os.path.exists(out_dir_tb):
            os.makedirs(out_dir_tb)
        if not os.path.exists(out_dir_eval):
            os.makedirs(out_dir_eval)
        if not os.path.exists(out_dir_vis):
            os.makedirs(out_dir_vis)
    accelerator.wait_for_everyone()

    # -------------------- Logging settings --------------------
    # Only the main process configures the log file; other ranks just print
    # to the console (avoids missing-folder races on per-rank timestamps).
    if accelerator.is_main_process:
        config_logging(cfg.logging, out_dir=out_dir_run)
    else:
        logging.basicConfig(level=logging.INFO)
    if accelerator.is_main_process:
        logging.info(f"start at {t_start}")
        logging.debug(f"args: {args}")
        logging.debug(f"config: {cfg}")
        logging.debug(
            f"accelerator: {accelerator.mixed_precision = }, {accelerator.split_batches = }"
        )
        assert accelerator.split_batches == args.split_batch

    # Initialize wandb
    if accelerator.is_main_process:
        if not args.no_wandb:
            if resume_run is not None:
                wandb_id = load_wandb_job_id(out_dir_run)
                wandb_cfg_dic = {
                    "id": wandb_id,
                    "resume": "must",
                    **cfg.wandb,
                }
            else:
                wandb_cfg_dic = {
                    "config": dict(cfg),
                    "name": job_name,
                    "mode": "online",
                    **cfg.wandb,
                }
            wandb_cfg_dic.update({"dir": out_dir_run})
            wandb_run = init_wandb(enable=True, **wandb_cfg_dic)
            save_wandb_job_id(wandb_run, out_dir_run)
        else:
            init_wandb(enable=False)

        # Tensorboard (should be initialized after wandb)
        tb_logger.set_dir(out_dir_tb)

        log_slurm_job_id(step=0)
    accelerator.wait_for_everyone()

    # -------------------- Device --------------------
    device = accelerator.device
    device_id = device.index if device.type == "cuda" else None
    device_id = 0 if device_id is None else device_id
    logging.info(f"{device = }, {device_id = }")
    n_gpu = accelerator.state.num_processes
    if accelerator.is_main_process:
        mixed_precision = accelerator.mixed_precision
        logging.info(f"{mixed_precision = }")
        logging.info(f"{n_gpu = }")

    # -------------------- Snapshot of code and config --------------------
    if resume_run is None:
        if accelerator.is_main_process:
            _output_path = os.path.join(out_dir_run, "config.yaml")
            with open(_output_path, "w+") as f:
                OmegaConf.save(config=cfg, f=f)
            logging.info(f"Config saved to {_output_path}")
            # Copy and archive code on the first run
            code_snapshot_path = os.path.join(out_dir_run, "code_snapshot.tar.gz")
            logging.info("Saving code snapshot...")
            create_code_snapshot(code_snapshot_path, source_dir=".")
            logging.info(f"Code snapshot saved to: {code_snapshot_path}")
    accelerator.wait_for_everyone()

    # -------------------- Gradient accumulation steps --------------------
    eff_bs = cfg.dataloader.effective_batch_size
    accumulation_steps = eff_bs / cfg.dataloader.max_train_batch_size / n_gpu
    if args.split_batch:
        assert cfg.dataloader.max_train_batch_size >= n_gpu, (
            "not enough batch size to split"
        )
        assert 0 == cfg.dataloader.max_train_batch_size % n_gpu, (
            f"can't split {cfg.dataloader.max_train_batch_size} into {n_gpu} gpus"
        )
        accumulation_steps = eff_bs / cfg.dataloader.max_train_batch_size
    else:
        accumulation_steps = eff_bs / cfg.dataloader.max_train_batch_size / n_gpu
    assert int(accumulation_steps) == accumulation_steps
    accumulation_steps = int(accumulation_steps)

    if accelerator.is_main_process:
        logging.info(
            f"Effective batch size: {eff_bs}, accumulation steps: {accumulation_steps}, on {n_gpu} devices"
        )

    # -------------------- Data --------------------
    init_loader_seed = cfg.dataloader.seed

    if init_loader_seed is None:
        loader_generator = None
    else:
        init_loader_seed += 1234321 * device_id  # account for multi-GPU randomness
        loader_generator = torch.Generator().manual_seed(init_loader_seed)

    # Training dataset
    cfg_train_data = OmegaConf.load(cfg.dataset_cfg.train)
    train_dataset: BaseTwoModalDataset = get_multi_frame_dataset(
        cfg_train_data,
        base_data_dir=base_data_dir,
        mode=DatasetMode.TRAIN,
        augmentation_args=cfg.augmentation,
        init_seed=init_loader_seed,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.dataloader.max_train_batch_size,
        num_workers=cfg.dataloader.num_workers,
        shuffle=True,
        generator=loader_generator,
        drop_last=True,
    )

    # -------------------- Model --------------------
    model = VideoFusion(model_config=cfg)

    # -------------------- Trainer --------------------
    from src.trainer.ivf_trainer import IVFTrainer
    trainer = IVFTrainer(
        cfg=cfg,
        model=model,
        train_dataloader=train_loader,
        accelerator=accelerator,
        out_dir_ckpt=out_dir_ckpt,
        out_dir_eval=out_dir_eval,
        out_dir_vis=out_dir_vis,
        accumulation_steps=accumulation_steps,
        n_gpu=n_gpu,
    )

    # -------------------- Checkpoint --------------------
    if resume_run is not None and accelerator.is_main_process:
        trainer.load_checkpoint(resume_run, load_trainer_state=False, resume_lr_scheduler=False)

    # -------------------- Training & Evaluation Loop --------------------
    accelerator.wait_for_everyone()
    try:
        with accelerator.autocast():
            trainer.train()
    except Exception as e:
        logging.exception(e)