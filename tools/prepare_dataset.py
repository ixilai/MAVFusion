# MAVFusion: one-shot dataset preparation tool
# Authors: Xilai Li, Weijun Jiang, Xiaosong Li, Yang Liu, Hongbin Wang, Tao Ye, Huafeng Li, Haishu Tan (ECCV 2026)
#
# Walks a directory of paired multi-modal video frames, generates per-sequence
# CSVs, a train/test split.json, and a ready-to-use dataset YAML. After this
# tool runs, the user can point train.py / test.py at the generated config.
#
# Expected input layout:
#   <source>/<seq_name>/<modality_a>/<frame>.jpg
#   <source>/<seq_name>/<modality_b>/<frame>.jpg
#   ...
#
# Examples:
#   python tools/prepare_dataset.py --source /data/my_videos --dataset-name MyDB
#   python tools/prepare_dataset.py --source /data/my_videos --modality-dirs ir vi --train-ratio 0.9
#   python tools/prepare_dataset.py --source /data/my_videos --dataset-name MyDB --force

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Project root is the parent of tools/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _collect_frames_seqfirst(
    seq_dir: Path,
    modality_dirs: Tuple[str, str],
    file_ext: str,
) -> Tuple[List[str], List[str], List[str]]:
    """Sequence-first layout: ``<root>/<seq>/<mod_a>/<frame>``.

    Returns (warnings, frames_a, frames_b) where frame paths are relative to
    the sequence directory in the form ``<mod>/<filename>``.
    """
    mod_a, mod_b = modality_dirs
    dir_a = seq_dir / mod_a
    dir_b = seq_dir / mod_b
    warnings: List[str] = []
    if not dir_a.is_dir():
        return [f"missing modality dir `{mod_a}`"], [], []
    if not dir_b.is_dir():
        return [f"missing modality dir `{mod_b}`"], [], []
    file_exts = {file_ext, file_ext.upper(), file_ext.lower()}
    files_a = sorted(p.name for p in dir_a.iterdir() if p.is_file() and p.suffix in file_exts)
    files_b = sorted(p.name for p in dir_b.iterdir() if p.is_file() and p.suffix in file_exts)
    if not files_a:
        warnings.append(f"no `{file_ext}` frames in `{mod_a}/`")
    if not files_b:
        warnings.append(f"no `{file_ext}` frames in `{mod_b}/`")
    if len(files_a) != len(files_b):
        warnings.append(
            f"frame count mismatch ({mod_a}={len(files_a)} vs {mod_b}={len(files_b)}); "
            f"truncating to the shorter side"
        )
        n = min(len(files_a), len(files_b))
        files_a = files_a[:n]
        files_b = files_b[:n]
    if files_a != files_b:
        warnings.append(
            "frame filenames do not match between modalities; pairing by sorted order"
        )
    return warnings, [f"{mod_a}/{n}" for n in files_a], [f"{mod_b}/{n}" for n in files_b]


def _collect_frames_modfirst(
    source_root: Path,
    seq_name: str,
    modality_dirs: Tuple[str, str],
    file_ext: str,
) -> Tuple[List[str], List[str], List[str]]:
    """Modality-first layout: ``<root>/<mod>/<seq>/<frame>``.

    The CSV format is identical to seq-first; only the on-disk walk differs.
    """
    mod_a, mod_b = modality_dirs
    dir_a = source_root / mod_a / seq_name
    dir_b = source_root / mod_b / seq_name
    warnings: List[str] = []
    if not dir_a.is_dir():
        return [f"missing modality dir `{mod_a}/{seq_name}`"], [], []
    if not dir_b.is_dir():
        return [f"missing modality dir `{mod_b}/{seq_name}`"], [], []
    file_exts = {file_ext, file_ext.upper(), file_ext.lower()}
    files_a = sorted(p.name for p in dir_a.iterdir() if p.is_file() and p.suffix in file_exts)
    files_b = sorted(p.name for p in dir_b.iterdir() if p.is_file() and p.suffix in file_exts)
    if not files_a:
        warnings.append(f"no `{file_ext}` frames in `{mod_a}/{seq_name}/`")
    if not files_b:
        warnings.append(f"no `{file_ext}` frames in `{mod_b}/{seq_name}/`")
    if len(files_a) != len(files_b):
        warnings.append(
            f"frame count mismatch ({mod_a}={len(files_a)} vs {mod_b}={len(files_b)}); "
            f"truncating to the shorter side"
        )
        n = min(len(files_a), len(files_b))
        files_a = files_a[:n]
        files_b = files_b[:n]
    if files_a != files_b:
        warnings.append("frame filenames do not match between modalities; pairing by sorted order")
    return warnings, [f"{mod_a}/{n}" for n in files_a], [f"{mod_b}/{n}" for n in files_b]


def _detect_layout(
    source_path: Path,
    modality_dirs: Tuple[str, str],
) -> str:
    """Return ``"mod-first"`` if modality subdirs are at the top level of
    ``source_path``; otherwise ``"seq-first"``.

    Detection rule: a child dir whose name matches one of the modality names
    (or common aliases) is treated as a modality directory. If at least one
    such dir exists at the top level, the layout is mod-first.
    """
    aliases = {
        "infrared", "visible",
        "ir", "vi", "rgb", "thermal",
        "t", "v", "i", "r",
    }
    if not modality_dirs:
        return "seq-first"
    for child in source_path.iterdir():
        if child.is_dir() and child.name.lower() in aliases:
            return "mod-first"
    return "seq-first"


def _split_train_test(
    seq_names: List[str],
    train_ratio: float,
    split_seed: int,
    explicit_test: Optional[List[str]] = None,
    explicit_train: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Randomly assign sequences to train/test, honoring explicit overrides."""
    import random

    explicit_test_set = set(explicit_test or [])
    explicit_train_set = set(explicit_train or [])

    overlap = explicit_test_set & explicit_train_set
    if overlap:
        raise ValueError(
            f"Sequences in both --explicit-test and --explicit-train: {sorted(overlap)}"
        )

    unknown_explicit = (explicit_test_set | explicit_train_set) - set(seq_names)
    if unknown_explicit:
        raise ValueError(
            f"Explicit sequence names not found in dataset: {sorted(unknown_explicit)}"
        )

    rng = random.Random(split_seed)
    shuffled = list(seq_names)
    rng.shuffle(shuffled)

    train: List[str] = list(explicit_train_set)
    test: List[str] = list(explicit_test_set)

    for seq in shuffled:
        if seq in explicit_train_set or seq in explicit_test_set:
            continue
        if seq in train or seq in test:
            continue
        if len(train) / max(len(train) + len(test), 1) < train_ratio:
            train.append(seq)
        else:
            test.append(seq)

    train.sort()
    test.sort()
    return train, test


def _existing_outputs(out_csv_dir: Path, dataset_name: str, force: bool) -> bool:
    """Return True if prep should skip (files exist and --force not set)."""
    split_path = out_csv_dir / "split.json"
    if split_path.exists() and not force:
        print(
            f"[skip] {split_path} already exists. Use --force to overwrite.",
            file=sys.stderr,
        )
        return True
    return False


def _write_sequence_csvs(
    seq_names: List[str],
    modality_dirs: Tuple[str, str],
    file_ext: str,
    out_csv_dir: Path,
    layout: str,
    source_root: Path,
) -> List[str]:
    """Write one ``<seq_name>.csv`` per sequence; return collected warnings.

    Dispatches to the layout-specific frame collector. The CSV format is the
    same regardless of layout (column 1: ``<mod_a>/<frame>``, column 2:
    ``<mod_b>/<frame>``); only the on-disk walk differs.
    """
    all_warnings: List[str] = []
    out_csv_dir.mkdir(parents=True, exist_ok=True)

    for seq_name in sorted(seq_names):
        if layout == "mod-first":
            warnings, frames_a, frames_b = _collect_frames_modfirst(
                source_root, seq_name, modality_dirs, file_ext
            )
        else:
            seq_dir = source_root / seq_name
            warnings, frames_a, frames_b = _collect_frames_seqfirst(
                seq_dir, modality_dirs, file_ext
            )
        for w in warnings:
            all_warnings.append(f"[{seq_name}] {w}")

        if not frames_a or not frames_b:
            print(f"[warn] {seq_name}: no frames written (see warnings above)", file=sys.stderr)
            continue

        csv_path = out_csv_dir / f"{seq_name}.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("ir,rgb\n")
            for a, b in zip(frames_a, frames_b):
                f.write(f"{a},{b}\n")

    return all_warnings


def _write_split_json(
    out_csv_dir: Path,
    train_seqs: List[str],
    test_seqs: List[str],
) -> Path:
    split_path = out_csv_dir / "split.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({"train": train_seqs, "test": test_seqs}, f, indent=2)
    return split_path


def _write_dataset_yaml(
    out_cfg_dir: Path,
    dataset_name: str,
    task_name: str,
    csv_dir_rel: str,
    layout: str = "seq-first",
    modality_dirs: Tuple[str, str] = ("infrared", "visible"),
) -> Path:
    out_cfg_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = out_cfg_dir / f"{dataset_name}_5-frame.yaml"
    extra_lines = ""
    if layout == "mod-first":
        # Force auto-discover mode so the dataset walks the on-disk mod-first
        # tree directly; the per-sequence CSVs (which are written in seq-first
        # path format) are still produced for inspection but ignored at
        # load time.
        extra_lines = (
            f"auto_discover: true\n"
            f"layout: mod-first\n"
            f"modality_subdirs: [{modality_dirs[0]}, {modality_dirs[1]}]\n"
        )
    yaml_path.write_text(
        f"""# Auto-generated by tools/prepare_dataset.py
class_name: m3svd_dataset  # any *_dataset key in src/dataset/__init__.py maps to BaseRGBIRDataset
disp_name: {task_name}-{dataset_name}
dir: {dataset_name}
csv_dir: {csv_dir_rel}
num_frames: 5
frame_gap_ls: [0]
stride: 1
frame_padding: true
{extra_lines}""",
        encoding="utf-8",
    )
    return yaml_path


def _write_prepared_md(
    out_csv_dir: Path,
    *,
    source: str,
    task_name: str,
    dataset_name: str,
    layout: str,
    modality_dirs: Tuple[str, str],
    file_ext: str,
    train_ratio: float,
    split_seed: int,
    train_seqs: List[str],
    test_seqs: List[str],
    warnings: List[str],
    yaml_path: Path,
    explicit_test: Optional[List[str]],
    explicit_train: Optional[List[str]],
) -> Path:
    md_path = out_csv_dir / "PREPARED.md"
    lines = [
        f"# {task_name}-{dataset_name}",
        "",
        f"Generated by `tools/prepare_dataset.py` at {datetime.datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        f"- source: `{source}`",
        f"- layout: `{layout}`",
        f"- modality subdirs: `{modality_dirs[0]}/`, `{modality_dirs[1]}/`",
        f"- file extension: `{file_ext}`",
        f"- train ratio: {train_ratio}",
        f"- split seed: {split_seed}",
    ]
    if explicit_test:
        lines.append(f"- explicit test sequences: {explicit_test}")
    if explicit_train:
        lines.append(f"- explicit train sequences: {explicit_train}")

    lines += [
        "",
        "## Outputs",
        f"- train sequences ({len(train_seqs)}): {train_seqs}",
        f"- test sequences ({len(test_seqs)}): {test_seqs}",
        f"- dataset YAML: `{yaml_path}`",
        "",
        "## Warnings",
    ]
    if warnings:
        lines += [f"- {w}" for w in warnings]
    else:
        lines.append("- (none)")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def run(
    source: str,
    task_name: str = "IVF",
    dataset_name: Optional[str] = None,
    modality_dirs: Tuple[str, str] = ("infrared", "visible"),
    train_ratio: float = 0.8,
    split_seed: int = 2025,
    file_ext: str = ".jpg",
    csv_root: str = "data_split",
    config_root: str = "config/dataset",
    explicit_test: Optional[List[str]] = None,
    explicit_train: Optional[List[str]] = None,
    force: bool = False,
    layout: str = "auto",
) -> Dict[str, Path]:
    """Programmatic entry point. Returns a dict of generated file paths.

    Suitable for calling from train.py / test.py when --raw_data_dir is given.

    ``layout`` is one of ``"auto"``, ``"seq-first"`` (each sequence has its
    own subdir containing the two modality dirs) or ``"mod-first"`` (the two
    modality dirs sit at the top level and each contains all sequences).
    """
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_dir():
        raise NotADirectoryError(f"--source is not a directory: {source_path}")

    if dataset_name is None:
        dataset_name = source_path.name
    dataset_name = dataset_name.replace(" ", "_")

    out_csv_dir = (PROJECT_ROOT / csv_root / task_name / dataset_name).resolve()
    out_cfg_dir = (PROJECT_ROOT / config_root / task_name / dataset_name).resolve()

    if _existing_outputs(out_csv_dir, dataset_name, force):
        # Existing prep: still return the YAML path so train/test can proceed
        return {
            "yaml": out_cfg_dir / f"{dataset_name}_5-frame.yaml",
            "csv_dir": out_csv_dir,
            "split": out_csv_dir / "split.json",
        }

    if layout == "auto":
        layout = _detect_layout(source_path, modality_dirs)
    if layout not in ("seq-first", "mod-first"):
        raise ValueError(f"Unknown layout: {layout!r}")

    seq_names: List[str] = []
    if layout == "mod-first":
        mod_a, mod_b = modality_dirs
        if not (source_path / mod_a).is_dir() or not (source_path / mod_b).is_dir():
            raise FileNotFoundError(
                f"mod-first layout expected subdirs {mod_a}/ and {mod_b}/ under "
                f"{source_path}, found neither both"
            )
        seqs_a = {p.name for p in (source_path / mod_a).iterdir() if p.is_dir()}
        seqs_b = {p.name for p in (source_path / mod_b).iterdir() if p.is_dir()}
        # Sequences present in both modalities
        seq_names = sorted(seqs_a & seqs_b)
        missing_b = seqs_a - seqs_b
        missing_a = seqs_b - seqs_a
        if missing_a or missing_b:
            print(
                f"[prep][warn] sequences missing from one modality will be skipped: "
                f"only_in_{mod_a}={sorted(missing_b)}, only_in_{mod_b}={sorted(missing_a)}",
                file=sys.stderr,
            )
    else:
        for child in sorted(source_path.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            seq_names.append(child.name)

    if not seq_names:
        mod_a, mod_b = modality_dirs
        seq_first = (
            f"{source_path}/<seq_name>/{mod_a}/, <seq_name>/{mod_b}/*.jpg"
        )
        mod_first = (
            f"{source_path}/{mod_a}/<seq_name>/, {mod_b}/<seq_name>/*.jpg"
        )
        raise ValueError(
            f"No sequence subdirectories found under {source_path}. "
            f"Expected layout (seq-first): {seq_first} "
            f"or (mod-first): {mod_first}"
        )

    print(f"[prep] Layout: {layout}")
    print(f"[prep] Found {len(seq_names)} sequence(s) in {source_path}")
    print(f"[prep] Modality subdirs: {modality_dirs[0]}/, {modality_dirs[1]}/")

    warnings = _write_sequence_csvs(
        seq_names, modality_dirs, file_ext, out_csv_dir, layout, source_path
    )
    if warnings:
        print(f"[prep] {len(warnings)} warning(s) during scan (see PREPARED.md):")
        for w in warnings:
            print(f"   - {w}")

    train_seqs, test_seqs = _split_train_test(
        list(seq_names),
        train_ratio=train_ratio,
        split_seed=split_seed,
        explicit_test=explicit_test,
        explicit_train=explicit_train,
    )
    split_path = _write_split_json(out_csv_dir, train_seqs, test_seqs)
    csv_dir_rel = str(out_csv_dir.relative_to(PROJECT_ROOT))
    yaml_path = _write_dataset_yaml(
        out_cfg_dir, dataset_name, task_name, csv_dir_rel,
        layout=layout, modality_dirs=modality_dirs,
    )
    md_path = _write_prepared_md(
        out_csv_dir,
        source=str(source_path),
        task_name=task_name,
        dataset_name=dataset_name,
        layout=layout,
        modality_dirs=modality_dirs,
        file_ext=file_ext,
        train_ratio=train_ratio,
        split_seed=split_seed,
        train_seqs=train_seqs,
        test_seqs=test_seqs,
        warnings=warnings,
        yaml_path=yaml_path.relative_to(PROJECT_ROOT),
        explicit_test=explicit_test,
        explicit_train=explicit_train,
    )

    print(f"[prep] train: {len(train_seqs)} seq(s), test: {len(test_seqs)} seq(s)")
    print(f"[prep] CSV dir:      {out_csv_dir}")
    print(f"[prep] Split file:   {split_path}")
    print(f"[prep] Dataset YAML: {yaml_path}")
    print(f"[prep] Manifest:     {md_path}")

    return {
        "yaml": yaml_path,
        "csv_dir": out_csv_dir,
        "split": split_path,
        "manifest": md_path,
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare a multi-modal video frame dataset for MAVFusion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", required=True, help="Root directory of the raw frame dataset")
    p.add_argument("--task-name", default="IVF", help="Task name (used for output organization)")
    p.add_argument("--dataset-name", default=None, help="Dataset name (default: basename of --source)")
    p.add_argument(
        "--modality-dirs",
        nargs=2,
        default=["infrared", "visible"],
        metavar=("MOD_A", "MOD_B"),
        help="Names of the two modality subdirectories under each sequence",
    )
    p.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of sequences used for training")
    p.add_argument("--split-seed", type=int, default=2025, help="Random seed for train/test split")
    p.add_argument("--file-ext", default=".jpg", help="Image file extension to include")
    p.add_argument("--csv-root", default="data_split", help="Root for generated split/CSV files")
    p.add_argument("--config-root", default="config/dataset", help="Root for generated dataset YAML")
    p.add_argument(
        "--explicit-test",
        default=None,
        help="Comma-separated sequence names forced into the test split",
    )
    p.add_argument(
        "--explicit-train",
        default=None,
        help="Comma-separated sequence names forced into the train split",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing split.json / CSVs")
    p.add_argument(
        "--layout",
        default="auto",
        choices=["auto", "seq-first", "mod-first"],
        help=(
            "On-disk layout of the source directory. "
            "'seq-first' = <root>/<seq>/<mod>/<frame> (default MAVFusion convention). "
            "'mod-first' = <root>/<mod>/<seq>/<frame>. "
            "'auto' detects by checking for modality-named subdirs at the top level."
        ),
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    explicit_test = [s.strip() for s in (args.explicit_test or "").split(",") if s.strip()]
    explicit_train = [s.strip() for s in (args.explicit_train or "").split(",") if s.strip()]

    run(
        source=args.source,
        task_name=args.task_name,
        dataset_name=args.dataset_name,
        modality_dirs=tuple(args.modality_dirs),
        train_ratio=args.train_ratio,
        split_seed=args.split_seed,
        file_ext=args.file_ext,
        csv_root=args.csv_root,
        config_root=args.config_root,
        explicit_test=explicit_test,
        explicit_train=explicit_train,
        force=args.force,
        layout=args.layout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
