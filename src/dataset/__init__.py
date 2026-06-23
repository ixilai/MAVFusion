# MAVFusion: dataset factory
# Authors: Xilai Li, Weijun Jiang, Xiaosong Li, Yang Liu, Hongbin Wang, Tao Ye, Huafeng Li, Haishu Tan (ECCV 2026)

import os
from omegaconf import OmegaConf

from .base_two_modal_dataset import BaseTwoModalDataset, DatasetMode
from .base_RGBIR_dataset import BaseRGBIRDataset

dataset_name_class_dict = {
    "m3svd_dataset": BaseRGBIRDataset,
    "vtmot_dataset": BaseRGBIRDataset,
    "hdo_dataset": BaseRGBIRDataset,
}


def get_multi_frame_dataset(
    cfg_data_split: OmegaConf, base_data_dir: str, mode: DatasetMode, **kwargs
) -> BaseTwoModalDataset:
    cfg_dataset = OmegaConf.create(
        OmegaConf.to_container(cfg_data_split, resolve=True)
    )
    if cfg_dataset.get("class_name") not in dataset_name_class_dict:
        raise NotImplementedError(
            f"Unknown dataset class_name: {cfg_dataset.get('class_name')!r}. "
            f"Supported: {sorted(dataset_name_class_dict.keys())}"
        )
    dataset_class = dataset_name_class_dict[cfg_dataset.pop("class_name")]
    dataset = dataset_class(
        mode=mode,
        dataset_dir=os.path.join(base_data_dir, cfg_dataset.pop("dir")),
        **cfg_dataset,
        **kwargs,
    )
    return dataset
