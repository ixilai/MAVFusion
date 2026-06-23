# MAVFusion dataset base
# Authors: Xilai Li, Weijun Jiang, Xiaosong Li, Yang Liu, Hongbin Wang, Tao Ye, Huafeng Li, Haishu Tan (ECCV 2026)
# Original scaffolding: Zixiang Zhao et al. (UniVF, NeurIPS 2025 Spotlight).

import csv
import io
import json
import logging
import random
import zipfile
from enum import Enum
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
import cv2

class DatasetMode(Enum):
    TRAIN = "train"
    EVAL = "eval"
    TEST = "test"


def read_csv(filename, delimiter=","):
    with open(filename, "r", newline="") as f:
        csv_reader = csv.reader(f, delimiter=delimiter)
        header = next(csv_reader)
        content = [row for row in csv_reader if row]
    return header, content


class BaseTwoModalDataset(Dataset):
    SEQ_IDX_KEY = "seq_idx"
    FRAME_IDX_LS_KEY = "frame_idx_ls"
    DIR_SEP_SYMBOL = "^"

    def __init__(
        self,
        filename_col_names: Dict,
        mode: DatasetMode,
        csv_dir: str,
        dataset_dir: str,
        disp_name: str,
        # Moving window
        num_frames: int,
        frame_gap_ls: List[int],
        stride: int,
        init_seed: int = None,
        frame_padding: bool = False,
        # Preprocessing
        augmentation_args: dict = None,
        rgb_transform=lambda x: x / 255.0,  #  [0, 255] -> [0, 1],
        # Other
        split_filename="split.json",
        # Auto-discover mode (bypasses split.json / per-sequence CSVs)
        auto_discover: bool = False,
        modality_subdirs: tuple = ("infrared", "visible"),
        train_ratio: float = 0.8,
        split_seed: int = 2025,
        layout: str = "seq-first",
        **kwargs,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.filename_col_names = filename_col_names[self.mode]

        # Dataset info
        self.dataset_dir = Path(dataset_dir)
        self.csv_dir = Path(csv_dir) if csv_dir is not None else None
        del dataset_dir, csv_dir
        assert self.dataset_dir.exists(), (
            f"Dataset does not exist at: {self.dataset_dir}"
        )
        self.disp_name = disp_name
        self.auto_discover = auto_discover

        # For multi-thread random generator
        self.is_process_init = False
        self.worker_info = None
        self.worker_id = None
        self.init_seed = init_seed

        self.rgn: torch.Generator = None

        # Moving window
        self.n_frames_per_sample: int = num_frames
        self.frame_gap_ls = frame_gap_ls
        self.stride = stride
        self.frame_padding = frame_padding
        # Random sampling setting
        self.random_sample_cfg = kwargs.pop("random_sample", None)
        self.use_random_sample = self.random_sample_cfg is not None

        if len(kwargs) > 0:
            logging.warning(f"Unexpected kwargs: {kwargs.keys()}")

        logging.debug(
            f"Dataset info: `{self.disp_name = }`: {self.n_frames_per_sample = }, {self.frame_gap_ls = }, {self.stride = }"
        )

        self.rgb_transform = rgb_transform

        # Training augmentation settings
        self.augm_args = augmentation_args
        logging.debug(f"{self.augm_args = }")

        # Handler of zip dataset (not supported in auto-discover mode)
        self.zip_ref = None
        self.is_zip = (
            False if auto_discover else
            True if self.dataset_dir.is_file() and zipfile.is_zipfile(self.dataset_dir)
            else False
        )

        self.csv_header = None
        self.seq_rel_dir_ls_w_nframe = []
        self.seq_csv_body_ls = []

        if auto_discover:
            self._auto_discover_sequences(modality_subdirs, train_ratio, split_seed, layout)
        else:
            # Get scene list from split.json
            with open((self.csv_dir / split_filename)) as f:
                split_json = json.load(f)
            self.scene_name_ls = split_json[self.mode.value]

            # Load per-sequence CSVs
            for scene_name in self.scene_name_ls:
                csv_name = f"{scene_name}.csv"
                header, csv_body = read_csv((self.csv_dir / csv_name))
                if self.csv_header is None:
                    self.csv_header = header
                assert header == self.csv_header, "csv headers don't match"
                assert set(self.filename_col_names).issubset(set(header)), (
                    f"csv header doesn't contain required columns {self.filename_col_names}"
                )
                n_frame = len(csv_body)
                seq_name = Path(csv_name).stem
                seq_rel_dir = seq_name.replace(self.DIR_SEP_SYMBOL, "/")
                self.seq_rel_dir_ls_w_nframe.append([seq_rel_dir, n_frame])
                self.seq_csv_body_ls.append(csv_body)
            assert len(self.seq_rel_dir_ls_w_nframe) == len(self.seq_csv_body_ls)

        # Generate sub-sequence id
        if not self.use_random_sample:
            self.sub_seq_dict_ls = self._generate_subsequence_ids()

    def _auto_discover_sequences(
        self,
        modality_subdirs: tuple,
        train_ratio: float,
        split_seed: int,
        layout: str = "seq-first",
    ) -> None:
        """Walk ``self.dataset_dir`` to find sequences and build the in-memory
        equivalents of ``scene_name_ls``, ``seq_rel_dir_ls_w_nframe`` and
        ``seq_csv_body_ls`` (i.e. everything the rest of the class needs without
        ever touching ``split.json`` or per-sequence ``*.csv`` files).

        ``layout`` is one of ``"seq-first"`` (``<dataset_dir>/<seq>/<mod>/...``)
        or ``"mod-first"`` (``<dataset_dir>/<mod>/<seq>/...``).
        """
        assert len(self.filename_col_names) == 2, (
            "auto_discover mode currently supports only 2-modality datasets"
        )
        # IR / VIS col names are stable in MAVFusion; map discovered subdirs
        # to the column names the rest of the loader expects.
        ir_col, rgb_col = self.filename_col_names
        mod_a, mod_b = modality_subdirs

        file_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        def _scan_pair(dir_a: Path, dir_b: Path):
            files_a = sorted(
                p.name for p in dir_a.iterdir()
                if p.is_file() and p.suffix.lower() in file_exts
            )
            files_b = sorted(
                p.name for p in dir_b.iterdir()
                if p.is_file() and p.suffix.lower() in file_exts
            )
            return files_a, files_b

        all_seqs: List[str] = []
        seq_frames: Dict[str, Tuple[List[str], List[str]]] = {}

        if layout == "mod-first":
            root_a = self.dataset_dir / mod_a
            root_b = self.dataset_dir / mod_b
            if not root_a.is_dir() or not root_b.is_dir():
                raise FileNotFoundError(
                    f"auto_discover (mod-first) requires both `{mod_a}/` and "
                    f"`{mod_b}/` under {self.dataset_dir}"
                )
            seqs_a = {p.name for p in root_a.iterdir() if p.is_dir() and not p.name.startswith(".")}
            seqs_b = {p.name for p in root_b.iterdir() if p.is_dir() and not p.name.startswith(".")}
            seq_names = sorted(seqs_a & seqs_b)
            only_a = sorted(seqs_a - seqs_b)
            only_b = sorted(seqs_b - seqs_a)
            if only_a:
                logging.warning(
                    f"[auto-discover] mod-first: sequences missing from `{mod_b}/`: {only_a}"
                )
            if only_b:
                logging.warning(
                    f"[auto-discover] mod-first: sequences missing from `{mod_a}/`: {only_b}"
                )
            for seq_name in seq_names:
                dir_a = root_a / seq_name
                dir_b = root_b / seq_name
                files_a, files_b = _scan_pair(dir_a, dir_b)
                if not files_a or not files_b:
                    logging.debug(
                        f"[auto-discover] skipping {seq_name}: no images in "
                        f"`{mod_a}/{seq_name}/` or `{mod_b}/{seq_name}/`"
                    )
                    continue
                n = min(len(files_a), len(files_b))
                if len(files_a) != len(files_b):
                    logging.warning(
                        f"[auto-discover] {seq_name}: frame count mismatch "
                        f"({mod_a}={len(files_a)} vs {mod_b}={len(files_b)}); "
                        f"truncating to {n}"
                    )
                seq_frames[seq_name] = (
                    [f"{mod_a}/{seq_name}/{n}" for n in files_a[:n]],
                    [f"{mod_b}/{seq_name}/{n}" for n in files_b[:n]],
                )
                all_seqs.append(seq_name)
        else:
            for child in sorted(self.dataset_dir.iterdir()):
                if not child.is_dir() or child.name.startswith("."):
                    continue
                dir_a = child / mod_a
                dir_b = child / mod_b
                if not dir_a.is_dir() or not dir_b.is_dir():
                    logging.debug(
                        f"[auto-discover] skipping {child.name}: missing {mod_a}/ or {mod_b}/"
                    )
                    continue
                files_a, files_b = _scan_pair(dir_a, dir_b)
                if not files_a or not files_b:
                    logging.debug(
                        f"[auto-discover] skipping {child.name}: no images in {mod_a}/ or {mod_b}/"
                    )
                    continue
                n = min(len(files_a), len(files_b))
                if len(files_a) != len(files_b):
                    logging.warning(
                        f"[auto-discover] {child.name}: frame count mismatch "
                        f"({mod_a}={len(files_a)} vs {mod_b}={len(files_b)}); "
                        f"truncating to {n}"
                    )
                seq_frames[child.name] = (
                    [f"{mod_a}/{n}" for n in files_a[:n]],
                    [f"{mod_b}/{n}" for n in files_b[:n]],
                )
                all_seqs.append(child.name)

        if not all_seqs:
            if layout == "mod-first":
                raise FileNotFoundError(
                    f"auto_discover found no sequences under {self.dataset_dir}. "
                    f"Expected mod-first layout: <dataset_dir>/{mod_a}/<seq>/, "
                    f"{mod_b}/<seq>/*.jpg"
                )
            raise FileNotFoundError(
                f"auto_discover found no sequences under {self.dataset_dir}. "
                f"Expected layout: <dataset_dir>/<seq>/{mod_a}/, <seq>/{mod_b}/"
            )

        # Train/test split: must use the exact same algorithm as
        # tools/prepare_dataset.py:_split_train_test, otherwise the prep tool's
        # split.json and the in-memory auto-discover will pick different
        # sequences for the test set (which can silently route non-divisible
        # resolutions to the model).
        from tools.prepare_dataset import _split_train_test
        train_seqs, test_seqs = _split_train_test(
            all_seqs, train_ratio, split_seed,
        )
        self.scene_name_ls = train_seqs if self.mode == DatasetMode.TRAIN else test_seqs

        # Build csv_header and seq_csv_body_ls to mirror the CSV-mode layout.
        # For mod-first layout, ``seq_rel_dir`` must be empty so that
        # ``Path(seq_rel_dir) / seq_csv_body`` resolves to just
        # ``<mod>/<frame>`` (the dataset reader joins this with
        # ``self.dataset_dir`` to get ``<dataset_dir>/<mod>/<frame>``).
        self.csv_header = [ir_col, rgb_col]
        seq_rel_dir_value = "" if layout == "mod-first" else seq_name
        for seq_name in self.scene_name_ls:
            frames_a, frames_b = seq_frames[seq_name]
            body = list(zip(frames_a, frames_b))
            self.seq_rel_dir_ls_w_nframe.append([seq_rel_dir_value, len(body)])
            self.seq_csv_body_ls.append(body)

        logging.info(
            f"[auto-discover] {self.disp_name}: {len(self.scene_name_ls)} sequence(s) "
            f"for mode={self.mode.value} "
            f"(train={len(train_seqs)}, test={len(test_seqs)})"
        )

    def _generate_subsequence_ids(self):
        # Check window settings
        if self.n_frames_per_sample < 0:
            logging.info(
                "num_frames < 0. Will use all frames in the sequence. Ignoring `frame_gap_ls` and `stride`."
            )
            self.frame_gap_ls = [0]
            self.stride = 0
        else:
            assert min(self.frame_gap_ls) >= 0, (
                f"frame gap has to be >=0, found {min(self.frame_gap_ls) = }"
            )
            assert self.stride > 0, f"stride has to be > 0, found {self.stride}."

        sub_seq_dict_ls = []  # [{"seq_idx": int, "frame_idx_ls": [...]}, ...]
        for frame_gap in self.frame_gap_ls:
            frame_interval = frame_gap + 1
            for seq_id, line in enumerate(self.seq_rel_dir_ls_w_nframe):
                seq_rel_dir, n_frame = line[:2]
                n_frame = int(n_frame)

                if self.n_frames_per_sample > 0:
                    # Number of frames is specified
                    n_min_frame = frame_interval * (self.n_frames_per_sample - 1) + 1
                    # Check if frames are enough
                    if n_min_frame > n_frame:
                        logging.warning(
                            f"Not enough frames ({n_frame}) in sequence {seq_rel_dir}, min. {n_min_frame} required."
                        )
                        continue
                    # Generate sub-sequence ids
                    for i in range(0, n_frame - n_min_frame + 1, self.stride):
                        frame_id_ls = list(range(i, i + n_min_frame, frame_interval))
                        sub_seq_dict_ls.append(
                            {
                                self.SEQ_IDX_KEY: seq_id,
                                self.FRAME_IDX_LS_KEY: frame_id_ls,
                            }
                        )
                        assert max(frame_id_ls) < n_frame
                    # Repeat first and last frame
                    if self.frame_padding:
                        assert 1 == self.stride
                        assert 0 == (self.n_frames_per_sample + 1) % 2, (
                            "number of frames per sample should be odd when `frame_padding`"
                        )
                        assert self.n_frames_per_sample > 1, (
                            f"too few frames for `frame_padding` {self.n_frames_per_sample = }"
                        )
                        for n_repeat in range(
                            1, int((self.n_frames_per_sample + 1) / 2)
                        ):
                            # first frame
                            frame_id_ls = [0] * n_repeat
                            frame_id_ls.extend(
                                range(0, self.n_frames_per_sample - n_repeat)
                            )
                            sub_seq_dict_ls.insert(
                                0,
                                {
                                    self.SEQ_IDX_KEY: seq_id,
                                    self.FRAME_IDX_LS_KEY: frame_id_ls,
                                },
                            )
                            # last frame
                            frame_id_ls = list(
                                range(
                                    n_frame - (self.n_frames_per_sample - n_repeat),
                                    n_frame,
                                )
                            )
                            frame_id_ls.extend([n_frame - 1] * n_repeat)
                            sub_seq_dict_ls.append(
                                {
                                    self.SEQ_IDX_KEY: seq_id,
                                    self.FRAME_IDX_LS_KEY: frame_id_ls,
                                }
                            )
                else:
                    # Use all frames
                    sub_seq_dict_ls.append(
                        {
                            self.SEQ_IDX_KEY: seq_id,
                            self.FRAME_IDX_LS_KEY: list(range(0, n_frame)),
                        }
                    )

        return sub_seq_dict_ls

    def __len__(self):
        if self.use_random_sample:
            return self.random_sample_cfg.random_length
        else:
            return len(self.sub_seq_dict_ls)

    def _process_init_(self):
        # get worker info
        self.worker_info = torch.utils.data.get_worker_info()
        if self.worker_info is None:
            self.worker_id = 0
        else:
            self.worker_id = self.worker_info.id
        logging.debug(
            f"dataloader process initialized at {self.worker_id = }. worker_info: {self.worker_info}"
        )
        self.is_process_init = True

        # Set random seed for rgn
        if self.init_seed is not None:
            seed = (
                self.init_seed
                + 222000 * self.worker_id
                + 1100 * self.n_frames_per_sample
            )  # type: ignore
            self.rgn = torch.Generator().manual_seed(seed)
            logging.debug(
                f"Generator of '{self.disp_name}' is seeded at {self.worker_id = } with {seed = }"
            )

    def __getitem__(self, index):
        if not self.is_process_init:
            self._process_init_()

        rasters, other = self._get_data_item(index)
        if DatasetMode.TRAIN == self.mode:
            rasters = self._training_preprocess(rasters)

        # merge
        outputs = rasters
        outputs.update(other)
        return outputs

    def _get_data_item(self, index):
        data_path_ls_dict = self._get_data_path(index=index)

        rasters: Dict[str, List] = {}  # list of rastersW

        for col_name in self.filename_col_names:
            rasters[col_name] = self._load_rgb_data(data_path_ls_dict[col_name])

        other = {
            "index": index,
            "data_path_ls_dict": data_path_ls_dict,
            "dataset": self.disp_name,
        }

        return rasters, other

    def _get_data_path(self, index):
        sub_seq_dict = self.sub_seq_dict_ls[index]
        seq_idx = sub_seq_dict[self.SEQ_IDX_KEY]
        frame_idx_ls = sub_seq_dict[self.FRAME_IDX_LS_KEY]

        seq_filenames = self.seq_csv_body_ls[seq_idx]
        seq_rel_dir = self.seq_rel_dir_ls_w_nframe[seq_idx][0]
        result_dict = {}
        for i_col in range(len(self.filename_col_names)):
            col_name = self.filename_col_names[i_col]
            result_dict[col_name] = [
                str(Path(seq_rel_dir) / seq_filenames[idx][i_col])
                for idx in frame_idx_ls
            ]
        return result_dict

    def _load_rgb_data(self, rgb_rel_path_ls):
        # Read RGB data
        rgb_int_ls = []
        rgb_norm_ls = []
        for rgb_rel_path in rgb_rel_path_ls:
            rgb = self._read_single_image(rgb_rel_path)  # [rgb, H, W]
            rgb_norm = self.rgb_transform(rgb)
            rgb_int_ls.append(torch.from_numpy(rgb).int().unsqueeze(0))
            rgb_norm_ls.append(torch.from_numpy(rgb_norm).float().unsqueeze(0))

        # rgb_int = torch.concat(rgb_int_ls, dim=0)  # [N, rgb, H, W]
        rgb_norm = torch.concat(rgb_norm_ls, dim=0)  # [N, rgb, H, W]

        return rgb_norm

    def _read_single_image(self, img_rel_path) -> np.ndarray:
        if self.is_zip:
            image_to_read = self._read_from_zip(img_rel_path)
        else:
            image_to_read = self.dataset_dir / img_rel_path
        image = Image.open(image_to_read)  # [H, W, rgb]
        image = np.asarray(image)

        # image = cv2.imread(image_to_read).astype('float32') 
        # image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # image = np.transpose(image, (2, 0, 1)).astype(int)  # [rgb, H, W]
        # Check if the image is grayscale (i.e., has only one channel)
        if image.ndim == 2:  # Grayscale image
            image = np.stack(
                [image] * 3, axis=0
            )  # Convert to 3 channels by stacking the same image three times
        else:
            image = np.transpose(image, (2, 0, 1))  # [rgb, H, W]
        image = image.astype(int)  # Convert the image to integer type
        return image

    def _read_npy(self, npy_rel_path) -> np.ndarray:
        if self.is_zip:
            npy_to_read = self._read_from_zip(npy_rel_path)
        else:
            npy_to_read = self.dataset_dir, npy_rel_path
        image = np.load(npy_to_read)
        return image

    def _read_from_zip(self, rel_path: str) -> io.BytesIO:
        if self.zip_ref is None:
            self.zip_ref = zipfile.ZipFile(self.dataset_dir)
        file_data = self.zip_ref.read(rel_path)
        file_data = io.BytesIO(file_data)
        return file_data

    def _training_preprocess(self, rasters):
        # Augmentation
        if self.augm_args is not None:
            rasters = self._augment_data(rasters)

        return rasters

    def _augment_data(self, rasters_dict):
        # left-right flipping
        lr_flip_p = self.augm_args.lr_flip_p
        if random.random() < lr_flip_p:
            rasters_dict = {k: v.flip(-1) for k, v in rasters_dict.items()}

        # random crop
        crop_size = self.augm_args.random_crop_hw
        if crop_size is not None:
            _, _, h, w = rasters_dict[list(rasters_dict.keys())[0]].shape
            top = torch.randint(0, h - crop_size[0] + 1, (1,)).item()
            left = torch.randint(0, w - crop_size[1] + 1, (1,)).item()
            rasters_dict = {
                k: F.crop(img, top, left, crop_size[0], crop_size[1])
                for k, img in rasters_dict.items()
            }

        return rasters_dict

    def _close_zip(self):
        if hasattr(self, "zip_ref") and self.zip_ref is not None:
            self.zip_ref.close()
            self.zip_ref = None

    def __del__(self):
        self._close_zip()
