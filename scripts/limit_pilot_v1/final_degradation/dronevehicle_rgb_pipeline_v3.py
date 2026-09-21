#!/usr/bin/env python3
"""DroneVehicle RGB enhancement paired-data pipeline, version 3.

Source images and the manual-review CSV are never modified.  Valid RGB regions
are cropped into lossless GT PNG files, then paired with deterministic haze,
globally monotonic low light, and truncated additive-white-Gaussian-noise
(AWGN) degradations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps


APP_VERSION = "3.3.0"
SPLITS = ("train", "val", "test")
SEVERITY_ORDER = ("light", "medium", "heavy")
LUMA_WEIGHTS = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
RECIPE_FLAGS: dict[str, tuple[bool, bool, bool]] = {
    "noise": (False, False, True),
    "dark": (False, True, False),
    "haze": (True, False, False),
    "dark_noise": (False, True, True),
    "haze_noise": (True, False, True),
    "haze_dark": (True, True, False),
    "haze_dark_noise": (True, True, True),
}
RECIPE_ORDER = tuple(RECIPE_FLAGS)

CLEAN_MANIFEST_FIELDS = [
    "dataset_id",
    "split",
    "selection_index",
    "record_key",
    "image_id",
    "filename",
    "export_filename",
    "source_path",
    "gt_path",
    "file_sha256",
    "gt_file_sha256",
    "manual_label",
    "export_mode",
    "analysis_roi",
    "roi_source",
    "roi_left",
    "roi_top",
    "roi_right",
    "roi_bottom",
    "source_width",
    "source_height",
    "gt_width",
    "gt_height",
    "selection_seed",
    "min_image_id_gap",
    "app_version",
]

PAIR_MANIFEST_FIELDS = [
    "pair_index",
    "pair_id",
    "split",
    "record_key",
    "image_id",
    "recipe",
    "severity",
    "haze_level",
    "dark_level",
    "noise_level",
    "has_haze",
    "has_dark",
    "has_noise",
    "operation_order",
    "gt_path",
    "input_path",
    "rng_seed",
    "haze_airlight_r",
    "haze_airlight_g",
    "haze_airlight_b",
    "haze_transmission_low",
    "haze_transmission_high",
    "haze_grid_h",
    "haze_grid_w",
    "dark_model",
    "dark_gamma",
    "dark_target_luma_p50",
    "dark_source_luma_p50",
    "dark_max_source_p50_ratio",
    "dark_effective_target_luma_p50",
    "dark_highlight_cap",
    "dark_scale",
    "noise_distribution",
    "noise_mean_255",
    "noise_sigma_255",
    "noise_gaussian_sigma",
    "noise_truncate_sigma",
    "mean_absolute_change",
    "pair_source_gt_count",
    "pair_selected_gt_count",
    "pair_selection_seed",
    "pair_selection_sha256",
    "degradation_config_sha256",
    "gt_manifest_sha256",
    "generation_mode",
    "app_version",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def eprint(*values: object) -> None:
    print(*values, file=sys.stderr)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv_atomic(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def stable_score(seed: int, key: str) -> bytes:
    return hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()


def safe_component(value: str, fallback: str = "item") -> str:
    value = Path(value).name.strip() or fallback
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = value.rstrip(". ")
    return value or fallback


def numeric_image_id(row: dict[str, str]) -> int | None:
    raw = row.get("image_id", "").strip()
    match = re.search(r"\d+", raw)
    return int(match.group()) if match else None


def resolve_under_root(root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def load_config(config_path: Path, root_override: str | None = None) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)

    root_value = root_override or config.get("root")
    if not root_value:
        raise RuntimeError("配置缺少 root。")
    root = Path(root_value)

    config["_config_path"] = str(config_path.resolve())
    config["_root_path"] = root
    config["_manual_results_path"] = resolve_under_root(
        root, config["manual_results_csv"]
    )
    config["_dataset_output_path"] = resolve_under_root(
        root, config["dataset_output"]
    )
    validate_config(config)
    return config


def validate_range(name: str, value: Any, positive: bool = False) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise RuntimeError(f"{name} 必须是两个数构成的列表。")
    low, high = float(value[0]), float(value[1])
    if low > high:
        raise RuntimeError(f"{name} 下限不能大于上限。")
    if positive and low <= 0:
        raise RuntimeError(f"{name} 必须大于0。")
    return low, high


def validate_config(config: dict[str, Any]) -> None:
    gt_export = config.get("gt_export", {})
    if gt_export.get("mode") != "crop_png":
        raise RuntimeError("gt_export.mode 必须是 crop_png，确保白边不会进入训练与评价。")
    if str(gt_export.get("image_format", "")).lower() != "png":
        raise RuntimeError("gt_export.image_format 必须是 png，避免二次有损压缩。")
    fallback = gt_export.get("fallback_for_840x712")
    if (
        not isinstance(fallback, list)
        or len(fallback) != 4
        or not all(isinstance(value, (int, float)) for value in fallback)
    ):
        raise RuntimeError("gt_export.fallback_for_840x712 必须包含4个坐标。")

    targets = config.get("target_counts", {})
    for split in SPLITS:
        value = targets.get(split)
        if value is not None and int(value) < 0:
            raise RuntimeError(f"target_counts.{split} 不能为负数。")

    degradation = config.get("degradation", {})
    weights = degradation.get("recipe_weights", {})
    unknown = sorted(set(weights) - set(RECIPE_FLAGS))
    missing = sorted(set(RECIPE_FLAGS) - set(weights))
    if unknown or missing:
        raise RuntimeError(
            f"recipe_weights 类别不完整。缺少={missing}，未知={unknown}"
        )
    if sum(float(value) for value in weights.values()) <= 0:
        raise RuntimeError("recipe_weights 总和必须大于0。")

    severity_weights = degradation.get("severity_weights", {})
    if set(severity_weights) != set(SEVERITY_ORDER):
        raise RuntimeError("severity_weights 必须完整包含 light、medium、heavy。")
    if sum(float(value) for value in severity_weights.values()) <= 0:
        raise RuntimeError("severity_weights 总和必须大于0。")

    order = degradation.get("operation_order")
    if order != ["haze", "dark", "noise"]:
        raise RuntimeError(
            "operation_order 必须固定为 haze → dark → noise，"
            "以符合先大气成像、再曝光响应、最后传感器噪声的顺序。"
        )

    haze = degradation.get("haze", {})
    dark = degradation.get("dark", {})
    noise = degradation.get("noise", {})
    validate_range("haze.airlight", haze.get("airlight"))
    validate_range("haze.coarse_grid", haze.get("coarse_grid"), positive=True)
    if noise.get("distribution") != "gaussian":
        raise RuntimeError("主实验 noise.distribution 必须是 gaussian，不混入泊松噪声。")
    truncate_sigma = float(noise.get("truncate_sigma", 0))
    if not (2.0 <= truncate_sigma <= 5.0):
        raise RuntimeError("noise.truncate_sigma 应在2到5之间，推荐3。")
    if dark.get("model") != "global_monotonic_luminance":
        raise RuntimeError(
            "dark.model必须是global_monotonic_luminance，"
            "确保正式数据使用已审批的纯全局暗光曲线。"
        )
    if dark.get("preserve_chroma") is not True:
        raise RuntimeError("dark.preserve_chroma必须为true。")
    if bool(dark.get("streetlight_spots")) or bool(dark.get("spatial_light_field")):
        raise RuntimeError("纯暗光模型禁止路灯光斑和空间光场。")

    previous_target_low: float | None = None
    previous_cap_low: float | None = None
    previous_source_ratio: float | None = None
    for severity in SEVERITY_ORDER:
        haze_level = haze.get("levels", {}).get(severity, {})
        dark_level = dark.get("levels", {}).get(severity, {})
        noise_level = noise.get("levels", {}).get(severity, {})
        low_range = validate_range(
            f"haze.levels.{severity}.transmission_low",
            haze_level.get("transmission_low"),
        )
        high_range = validate_range(
            f"haze.levels.{severity}.transmission_high",
            haze_level.get("transmission_high"),
        )
        if not (0.0 < low_range[0] <= low_range[1] < 1.0):
            raise RuntimeError(f"{severity} 雾的 transmission_low 必须位于(0,1)。")
        if not (0.0 < high_range[0] <= high_range[1] <= 1.0):
            raise RuntimeError(f"{severity} 雾的 transmission_high 必须位于(0,1]。")
        gamma_range = validate_range(
            f"dark.levels.{severity}.gamma",
            dark_level.get("gamma"),
            positive=True,
        )
        target_range = validate_range(
            f"dark.levels.{severity}.target_luma_p50",
            dark_level.get("target_luma_p50"),
            positive=True,
        )
        cap_range = validate_range(
            f"dark.levels.{severity}.highlight_cap",
            dark_level.get("highlight_cap"),
            positive=True,
        )
        if not (1.0 <= gamma_range[0] <= gamma_range[1] <= 4.0):
            raise RuntimeError(f"{severity}暗光gamma必须位于[1,4]。")
        if not (0.0 < target_range[0] <= target_range[1] < 1.0):
            raise RuntimeError(f"{severity}目标亮度中位数必须位于(0,1)。")
        if not (0.0 < cap_range[0] <= cap_range[1] < 1.0):
            raise RuntimeError(f"{severity}高亮上限必须位于(0,1)。")
        if target_range[1] >= cap_range[0]:
            raise RuntimeError(f"{severity}目标亮度必须严格低于高亮上限。")
        if previous_target_low is not None and target_range[1] >= previous_target_low:
            raise RuntimeError("轻、中、重目标亮度范围必须互不重叠并严格递减。")
        if previous_cap_low is not None and cap_range[1] >= previous_cap_low:
            raise RuntimeError("轻、中、重高亮上限必须互不重叠并严格递减。")
        source_ratio = float(dark_level.get("max_source_p50_ratio", 0.0))
        if not (0.0 < source_ratio < 1.0):
            raise RuntimeError(
                f"{severity}.max_source_p50_ratio必须位于(0,1)。"
            )
        if previous_source_ratio is not None and source_ratio >= previous_source_ratio:
            raise RuntimeError("轻、中、重相对原图亮度上限必须严格递减。")
        previous_target_low = target_range[0]
        previous_cap_low = cap_range[0]
        previous_source_ratio = source_ratio
        validate_range(
            f"noise.levels.{severity}.sigma_255",
            noise_level.get("sigma_255"),
            positive=True,
        )

    policy = degradation.get("combination_policy", {})
    expected_lengths = {"single": 1, "double": 2, "triple": 3}
    for group_name, expected_length in expected_lengths.items():
        group = policy.get(group_name, {})
        for severity in SEVERITY_ORDER:
            levels = group.get(severity)
            if not isinstance(levels, list) or len(levels) != expected_length:
                raise RuntimeError(
                    f"combination_policy.{group_name}.{severity} "
                    f"必须含{expected_length}个分量等级。"
                )
            if any(level not in SEVERITY_ORDER for level in levels):
                raise RuntimeError("combination_policy 中出现未知等级。")


def manual_source_path(row: dict[str, str], root: Path) -> Path:
    for field in ("absolute_path", "source_path", "hardlink_path"):
        raw = row.get(field, "").strip()
        if raw:
            candidate = Path(raw)
            if candidate.is_file():
                return candidate
    relative = row.get("relative_path", "").strip()
    if relative:
        for base in (root / "01_raw", root):
            candidate = base / Path(relative)
            if candidate.is_file():
                return candidate
    raw = row.get("absolute_path", "").strip()
    return Path(raw) if raw else root / "__missing_source__"


def load_manual_clean_candidates(config: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    path: Path = config["_manual_results_path"]
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到人工结果表：{path}\n"
            "请先用四键筛选工具完成审核。"
        )
    fields, rows = read_csv(path)
    required = {"record_key", "split", "manual_label", "image_id", "filename"}
    missing = sorted(required - set(fields))
    if missing:
        raise RuntimeError(f"人工结果表缺少列：{', '.join(missing)}")

    clean_label = str(config.get("manual_clean_label", "day_clean"))
    result: dict[str, list[dict[str, str]]] = {split: [] for split in SPLITS}
    missing_sources: list[str] = []
    for row in rows:
        split = row.get("split", "").strip()
        if split not in result or row.get("manual_label", "").strip() != clean_label:
            continue
        source = manual_source_path(row, config["_root_path"])
        if not source.is_file():
            missing_sources.append(str(source))
            continue
        copied = dict(row)
        copied["_source_path"] = str(source)
        result[split].append(copied)

    if missing_sources:
        preview = "\n".join(f"  {item}" for item in missing_sources[:10])
        raise RuntimeError(
            f"有{len(missing_sources)}条人工干净记录找不到原图，示例：\n{preview}"
        )
    return result


def choose_rows(
    candidates: list[dict[str, str]],
    target_count: int | None,
    seed: int,
    min_gap: int,
    forbidden_hashes: set[str],
) -> list[dict[str, str]]:
    ordered = sorted(
        candidates,
        key=lambda row: stable_score(seed, row.get("record_key", "")),
    )
    selected: list[dict[str, str]] = []
    selected_ids: list[int] = []
    local_hashes: set[str] = set()

    for row in ordered:
        file_hash = row.get("file_sha256", "").strip().lower()
        if file_hash and (file_hash in forbidden_hashes or file_hash in local_hashes):
            continue
        image_number = numeric_image_id(row)
        if min_gap > 0 and image_number is not None:
            if any(abs(image_number - old) < min_gap for old in selected_ids):
                continue
        selected.append(row)
        if image_number is not None:
            selected_ids.append(image_number)
        if file_hash:
            local_hashes.add(file_hash)
        if target_count is not None and len(selected) >= target_count:
            break

    if target_count is not None and len(selected) < target_count:
        raise RuntimeError(
            f"需要{target_count}张，但在去重和最小编号间隔={min_gap}后"
            f"只能选出{len(selected)}张。请继续人工筛选，或谨慎减小间隔。"
        )

    return sorted(
        selected,
        key=lambda row: (
            numeric_image_id(row) if numeric_image_id(row) is not None else 10**18,
            row.get("record_key", ""),
        ),
    )


def make_export_filename(
    row: dict[str, str],
    used: set[str],
) -> str:
    source = Path(row["_source_path"])
    source_name = row.get("filename", "").strip() or source.name
    stem = safe_component(Path(source_name).stem, "image")
    candidate = f"{stem}.png"
    if candidate.lower() not in used:
        used.add(candidate.lower())
        return candidate
    image_id = safe_component(row.get("image_id", ""), "image")
    candidate = f"{image_id}_{stem}.png"
    if candidate.lower() in used:
        digest = hashlib.sha256(row.get("record_key", "").encode("utf-8")).hexdigest()[:8]
        candidate = f"{image_id}_{digest}_{stem}.png"
    used.add(candidate.lower())
    return candidate


def parse_roi_text(
    roi_text: str,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    parts = [part.strip() for part in roi_text.split(",") if part.strip()]
    if len(parts) != 4:
        return None
    try:
        left, top, right, bottom = (int(part) for part in parts)
    except ValueError:
        return None
    if 0 <= left < right <= width and 0 <= top < bottom <= height:
        return left, top, right, bottom
    return None


def resolve_roi(
    row: dict[str, str],
    width: int,
    height: int,
    config: dict[str, Any],
) -> tuple[tuple[int, int, int, int], str]:
    gt_export = config["gt_export"]
    if bool(gt_export.get("use_analysis_roi", True)):
        parsed = parse_roi_text(row.get("analysis_roi", ""), width, height)
        if parsed is not None:
            return parsed, "analysis_roi"
    if (width, height) == (840, 712):
        fallback = tuple(int(value) for value in gt_export["fallback_for_840x712"])
        left, top, right, bottom = fallback
        if 0 <= left < right <= width and 0 <= top < bottom <= height:
            return (left, top, right, bottom), "fallback_840x712"
    return (0, 0, width, height), "full_image"


def export_cropped_gt(
    source: Path,
    target: Path,
    row: dict[str, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        opened.load()
        image = ImageOps.exif_transpose(opened).convert("RGB")
    source_width, source_height = image.size
    roi, roi_source = resolve_roi(row, source_width, source_height, config)
    cropped = image.crop(roi)
    temp_path = target.with_name(target.stem + ".tmp.png")
    cropped.save(temp_path, format="PNG", optimize=False)
    os.replace(temp_path, target)
    left, top, right, bottom = roi
    return {
        "analysis_roi": row.get("analysis_roi", ""),
        "roi_source": roi_source,
        "roi_left": left,
        "roi_top": top,
        "roi_right": right,
        "roi_bottom": bottom,
        "source_width": source_width,
        "source_height": source_height,
        "gt_width": cropped.width,
        "gt_height": cropped.height,
        "gt_file_sha256": sha256_file(target),
    }


def export_one(source: Path, target: Path, mode: str) -> None:
    """Legacy helper retained only for backward-readable imports."""
    if mode == "hardlink":
        try:
            os.link(source, target)
        except OSError as exc:
            raise RuntimeError(
                f"硬链接失败：{source} -> {target}\n"
                "请确认源图与输出目录都在同一个F盘NTFS卷。"
            ) from exc
    elif mode == "copy":
        shutil.copy2(source, target)
    else:
        raise RuntimeError(f"未知导出模式：{mode}")


def command_count(config: dict[str, Any], _args: argparse.Namespace) -> None:
    path: Path = config["_manual_results_path"]
    fields, rows = read_csv(path)
    if "manual_label" not in fields or "split" not in fields:
        raise RuntimeError("人工结果表缺少 split 或 manual_label。")
    counts = Counter(
        (row.get("split", ""), row.get("manual_label", ""))
        for row in rows
    )
    print("========== 人工筛选数量 ==========")
    for split in SPLITS:
        day = counts[(split, "day_clean")]
        night = counts[(split, "twilight_night_clean")]
        rejected = counts[(split, "not_clean")]
        pending = counts[(split, "pending")]
        total = day + night + rejected + pending
        print(
            f"{split:5s}  白天干净={day:5d}  黄昏/黑夜干净={night:5d}  "
            f"不干净={rejected:5d}  待定={pending:5d}  已审核={total:5d}"
        )
    test_target = int(config.get("target_counts", {}).get("test") or 0)
    readiness_note = ""
    try:
        candidates = load_manual_clean_candidates(config)
        targets = config.get("target_counts", {})
        gaps = config.get("min_image_id_gap", {})
        seed = int(config.get("selection_seed", 20260810))
        forbidden_hashes: set[str] = set()
        for split in SPLITS:
            raw_target = targets.get(split)
            target = None if raw_target is None else int(raw_target)
            selected = [] if target == 0 else choose_rows(
                candidates[split],
                target,
                seed + SPLITS.index(split) * 1009,
                int(gaps.get(split, 0)),
                forbidden_hashes,
            )
            for row in selected:
                file_hash = row.get("file_sha256", "").strip().lower()
                if file_hash:
                    forbidden_hashes.add(file_hash)
        test_ready = True
    except Exception as exc:  # noqa: BLE001
        test_ready = False
        readiness_note = str(exc)
    print(f"测试GT目标：{test_target}；状态：{'READY' if test_ready else 'NOT READY'}")
    if readiness_note:
        print(f"原因：{readiness_note}")
    train_pair_raw = config.get("fixed_pair_target_counts", {}).get("train")
    if train_pair_raw is not None:
        train_pair_target = int(train_pair_raw)
        train_pair_ready = test_ready and len(candidates["train"]) >= train_pair_target
        print(
            f"固定训练配对目标：{train_pair_target}；"
            f"状态：{'READY' if train_pair_ready else 'NOT READY'}"
        )


def command_prepare(config: dict[str, Any], args: argparse.Namespace) -> None:
    output_root: Path = config["_dataset_output_path"]
    manifest_dir = output_root / "manifests"
    combined_path = manifest_dir / "clean_splits.csv"
    if combined_path.exists() and not getattr(args, "self_test_internal", False):
        raise RuntimeError(
            f"正式数据快照已存在：{combined_path}\n"
            "为保护可复现性，本工具不会覆盖。若要重新选样，请在config.json中"
            "把 dataset_output 改成新的版本目录。"
        )

    candidates = load_manual_clean_candidates(config)
    targets = config.get("target_counts", {})
    seed = int(config.get("selection_seed", 20260810))
    gaps = config.get("min_image_id_gap", {})
    mode = str(config.get("gt_export", {}).get("mode", "crop_png"))
    forbidden_hashes: set[str] = set()
    selected_by_split: dict[str, list[dict[str, str]]] = {}

    print("========== 干净GT候选 ==========")
    for split in SPLITS:
        raw_target = targets.get(split)
        target = None if raw_target is None else int(raw_target)
        if target == 0:
            selected: list[dict[str, str]] = []
        else:
            selected = choose_rows(
                candidates[split],
                target,
                seed + SPLITS.index(split) * 1009,
                int(gaps.get(split, 0)),
                forbidden_hashes,
            )
        selected_by_split[split] = selected
        for row in selected:
            file_hash = row.get("file_sha256", "").strip().lower()
            if file_hash:
                forbidden_hashes.add(file_hash)
        target_text = "全部" if target is None else str(target)
        print(
            f"{split:5s} 候选={len(candidates[split]):5d}  "
            f"目标={target_text:>5s}  入选={len(selected):5d}"
        )

    if getattr(args, "dry_run", False):
        print("Dry-run完成；未创建目录、裁剪GT或清单。")
        return

    all_manifest_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        split_rows: list[dict[str, Any]] = []
        used_names: set[str] = set()
        gt_dir = output_root / "clean_gt" / split
        for index, row in enumerate(selected_by_split[split], start=1):
            source = Path(row["_source_path"])
            export_filename = make_export_filename(row, used_names)
            gt_path = gt_dir / export_filename
            crop_metadata = export_cropped_gt(source, gt_path, row, config)
            source_hash = row.get("file_sha256", "").strip() or sha256_file(source)
            item = {
                "dataset_id": "DroneVehicle_RGB_enhancement_v3",
                "split": split,
                "selection_index": index,
                "record_key": row.get("record_key", ""),
                "image_id": row.get("image_id", ""),
                "filename": row.get("filename", source.name),
                "export_filename": export_filename,
                "source_path": str(source),
                "gt_path": str(gt_path),
                "file_sha256": source_hash,
                "gt_file_sha256": crop_metadata["gt_file_sha256"],
                "manual_label": row.get("manual_label", ""),
                "export_mode": mode,
                "analysis_roi": crop_metadata["analysis_roi"],
                "roi_source": crop_metadata["roi_source"],
                "roi_left": crop_metadata["roi_left"],
                "roi_top": crop_metadata["roi_top"],
                "roi_right": crop_metadata["roi_right"],
                "roi_bottom": crop_metadata["roi_bottom"],
                "source_width": crop_metadata["source_width"],
                "source_height": crop_metadata["source_height"],
                "gt_width": crop_metadata["gt_width"],
                "gt_height": crop_metadata["gt_height"],
                "selection_seed": seed,
                "min_image_id_gap": int(gaps.get(split, 0)),
                "app_version": APP_VERSION,
            }
            split_rows.append(item)
            all_manifest_rows.append(item)
        write_csv_atomic(
            manifest_dir / f"{split}_gt.csv",
            CLEAN_MANIFEST_FIELDS,
            split_rows,
        )

    write_csv_atomic(combined_path, CLEAN_MANIFEST_FIELDS, all_manifest_rows)
    write_json_atomic(
        manifest_dir / "selection_metadata.json",
        {
            "created_at_utc": now_utc(),
            "app_version": APP_VERSION,
            "root": str(config["_root_path"]),
            "manual_results_csv": str(config["_manual_results_path"]),
            "dataset_output": str(output_root),
            "manual_clean_label": config.get("manual_clean_label"),
            "target_counts": config.get("target_counts"),
            "actual_counts": {
                split: len(selected_by_split[split]) for split in SPLITS
            },
            "selection_seed": seed,
            "min_image_id_gap": gaps,
            "gt_export": config.get("gt_export"),
            "source_manual_csv_sha256": sha256_file(config["_manual_results_path"]),
        },
    )
    print(f"正式干净GT快照已建立：{output_root}")
    print("原图未移动、未改写；GT已按有效ROI裁剪并保存为无损PNG。")


def load_rgb_float(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image, dtype=np.float32) / 255.0


def save_rgb_float(path: Path, array: np.ndarray, image_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(
        np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8),
        mode="RGB",
    )
    temp = path.with_name(path.stem + ".tmp" + path.suffix)
    fmt = image_format.lower()
    if fmt == "png":
        image.save(temp, format="PNG", optimize=False)
    elif fmt in {"jpg", "jpeg"}:
        image.save(temp, format="JPEG", quality=95, subsampling=0)
    else:
        raise RuntimeError(f"不支持的image_format：{image_format}")
    os.replace(temp, path)


def sample_uniform(rng: np.random.Generator, bounds: Sequence[float]) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def sample_degradation_parameters(
    rng: np.random.Generator,
    recipe: str,
    severity: str,
    degradation: dict[str, Any],
) -> dict[str, Any]:
    has_haze, has_dark, has_noise = RECIPE_FLAGS[recipe]
    if severity not in SEVERITY_ORDER:
        raise RuntimeError(f"未知退化等级：{severity}")
    components = [
        name
        for name, enabled in zip(
            ("haze", "dark", "noise"),
            (has_haze, has_dark, has_noise),
        )
        if enabled
    ]
    group_name = {1: "single", 2: "double", 3: "triple"}[len(components)]
    component_levels = list(
        degradation["combination_policy"][group_name][severity]
    )
    rng.shuffle(component_levels)
    assigned_levels = dict(zip(components, component_levels))
    params: dict[str, Any] = {
        "recipe": recipe,
        "severity": severity,
        "has_haze": has_haze,
        "has_dark": has_dark,
        "has_noise": has_noise,
        "haze_level": assigned_levels.get("haze", ""),
        "dark_level": assigned_levels.get("dark", ""),
        "noise_level": assigned_levels.get("noise", ""),
    }
    if has_haze:
        haze = degradation["haze"]
        haze_level = haze["levels"][assigned_levels["haze"]]
        air_base = sample_uniform(rng, haze["airlight"])
        air_jitter = rng.uniform(-0.025, 0.025, size=3)
        params["airlight"] = np.clip(air_base + air_jitter, 0.80, 1.0).tolist()
        t_low = sample_uniform(rng, haze_level["transmission_low"])
        t_high = sample_uniform(rng, haze_level["transmission_high"])
        if t_high <= t_low + 0.05:
            t_high = min(0.98, t_low + 0.05)
        grid_low, grid_high = (int(v) for v in haze["coarse_grid"])
        params["transmission_low"] = t_low
        params["transmission_high"] = t_high
        params["grid_h"] = int(rng.integers(grid_low, grid_high + 1))
        params["grid_w"] = int(rng.integers(grid_low, grid_high + 1))
    if has_dark:
        dark = degradation["dark"]
        dark_level = dark["levels"][assigned_levels["dark"]]
        params["dark_model"] = dark["model"]
        params["gamma"] = sample_uniform(rng, dark_level["gamma"])
        params["target_luma_p50"] = sample_uniform(
            rng, dark_level["target_luma_p50"]
        )
        params["max_source_p50_ratio"] = float(
            dark_level["max_source_p50_ratio"]
        )
        params["highlight_cap"] = sample_uniform(
            rng, dark_level["highlight_cap"]
        )
    if has_noise:
        noise = degradation["noise"]
        noise_level = noise["levels"][assigned_levels["noise"]]
        sigma_255 = sample_uniform(rng, noise_level["sigma_255"])
        params["noise_distribution"] = "gaussian"
        params["noise_mean_255"] = float(noise.get("mean_255", 0.0))
        params["noise_sigma_255"] = sigma_255
        params["gaussian_sigma"] = sigma_255 / 255.0
        params["noise_truncate_sigma"] = float(noise["truncate_sigma"])
    return params


def apply_haze(
    image: np.ndarray,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = image.shape[:2]
    coarse = rng.random((params["grid_h"], params["grid_w"]), dtype=np.float32)
    field_image = Image.fromarray(coarse, mode="F").resize(
        (width, height),
        resample=Image.Resampling.BICUBIC,
    )
    field = np.asarray(field_image, dtype=np.float32).copy()
    field -= float(field.min())
    peak = float(field.max())
    if peak > 1e-8:
        field /= peak
    t_low = float(params["transmission_low"])
    t_high = float(params["transmission_high"])
    transmission = t_low + (t_high - t_low) * field
    transmission = np.clip(transmission[..., None], 0.05, 1.0)
    airlight = np.asarray(params["airlight"], dtype=np.float32).reshape(1, 1, 3)
    return np.clip(image * transmission + airlight * (1.0 - transmission), 0.0, 1.0)


def rgb_luma(image: np.ndarray) -> np.ndarray:
    return np.sum(image.astype(np.float32) * LUMA_WEIGHTS, axis=2)


def apply_dark(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    if params.get("dark_model") != "global_monotonic_luminance":
        raise RuntimeError("未知或缺失的暗光模型。")
    gamma = float(params["gamma"])
    sampled_target_median = float(params["target_luma_p50"])
    max_source_ratio = float(params["max_source_p50_ratio"])
    highlight_cap = float(params["highlight_cap"])
    source = np.clip(image.astype(np.float32), 0.0, 1.0)
    source_y = rgb_luma(source)
    source_median = float(np.median(source_y))
    if source_median <= 1e-6:
        raise RuntimeError("暗光输入的有效亮度为零，无法保留车辆结构。")
    target_median = min(
        sampled_target_median,
        source_median * max_source_ratio,
    )
    score = np.power(np.maximum(source_y, 1e-6), gamma)

    # Solve one global scale so the transformed luminance reaches the sampled
    # target median.  No x/y coordinates, masks or detected objects are used.
    def curve(scale: float) -> np.ndarray:
        return highlight_cap * (
            1.0 - np.exp(-score * scale / highlight_cap)
        )

    low_scale = 0.0
    high_scale = 1.0
    while float(np.median(curve(high_scale))) < target_median:
        high_scale *= 2.0
        if high_scale > 1e8:
            raise RuntimeError("暗光目标亮度求解失败：输入图像有效亮度过低。")
    for _ in range(40):
        scale = 0.5 * (low_scale + high_scale)
        if float(np.median(curve(scale))) < target_median:
            low_scale = scale
        else:
            high_scale = scale
    scale = 0.5 * (low_scale + high_scale)
    target_y = curve(scale)

    # Multiplying every RGB channel by the same per-pixel luminance ratio
    # preserves chroma and geometry.  The response itself is global and
    # monotonic, so it cannot invent streetlights or local illumination blobs.
    output = source * (target_y / np.maximum(source_y, 1e-6))[..., None]
    # A degradation must never brighten an already-dark clean pixel.  This
    # guard matters for unusually dark daytime GTs whose median is below the
    # absolute target range.
    output = np.minimum(output, source)
    params["source_luma_p50"] = source_median
    params["effective_target_luma_p50"] = target_median
    params["dark_scale"] = scale
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def apply_noise(
    image: np.ndarray,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    sigma = float(params["gaussian_sigma"])
    mean = float(params.get("noise_mean_255", 0.0)) / 255.0
    truncate = float(params["noise_truncate_sigma"])
    noise = rng.normal(mean, sigma, size=image.shape).astype(np.float32)
    noise = np.clip(noise, mean - truncate * sigma, mean + truncate * sigma)
    return np.clip(image + noise, 0.0, 1.0)


def apply_degradation(
    clean: np.ndarray,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    output = clean.astype(np.float32, copy=True)
    if params["has_haze"]:
        output = apply_haze(output, params, rng)
    if params["has_dark"]:
        output = apply_dark(output, params)
    if params["has_noise"]:
        output = apply_noise(output, params, rng)
    return np.clip(output, 0.0, 1.0)


def exact_recipe_assignments(n: int, weights: dict[str, Any]) -> list[str]:
    total = sum(float(weights[name]) for name in RECIPE_ORDER)
    raw = {name: n * float(weights[name]) / total for name in RECIPE_ORDER}
    counts = {name: int(math.floor(raw[name])) for name in RECIPE_ORDER}
    remainder = n - sum(counts.values())
    ranked = sorted(
        RECIPE_ORDER,
        key=lambda name: (raw[name] - counts[name], -RECIPE_ORDER.index(name)),
        reverse=True,
    )
    for name in ranked[:remainder]:
        counts[name] += 1
    result: list[str] = []
    for name in RECIPE_ORDER:
        result.extend([name] * counts[name])
    return result


def exact_severity_assignments(n: int, weights: dict[str, Any]) -> list[str]:
    total = sum(float(weights[name]) for name in SEVERITY_ORDER)
    raw = {name: n * float(weights[name]) / total for name in SEVERITY_ORDER}
    counts = {name: int(math.floor(raw[name])) for name in SEVERITY_ORDER}
    remainder = n - sum(counts.values())
    ranked = sorted(
        SEVERITY_ORDER,
        key=lambda name: (raw[name] - counts[name], -SEVERITY_ORDER.index(name)),
        reverse=True,
    )
    for name in ranked[:remainder]:
        counts[name] += 1
    result: list[str] = []
    for name in SEVERITY_ORDER:
        result.extend([name] * counts[name])
    return result


def balanced_severity_allocations(
    recipe_counts: Counter[str],
    weights: dict[str, Any],
) -> dict[str, dict[str, int]]:
    total_items = sum(recipe_counts.values())
    global_targets = Counter(exact_severity_assignments(total_items, weights))
    weight_total = sum(float(weights[name]) for name in SEVERITY_ORDER)
    allocations: dict[str, dict[str, int]] = {}
    fractions: dict[str, dict[str, float]] = {}
    used = Counter()
    for recipe in RECIPE_ORDER:
        count = recipe_counts[recipe]
        allocations[recipe] = {}
        fractions[recipe] = {}
        for severity in SEVERITY_ORDER:
            raw = count * float(weights[severity]) / weight_total
            base = int(math.floor(raw))
            allocations[recipe][severity] = base
            fractions[recipe][severity] = raw - base
            used[severity] += base

    remaining = Counter(
        {
            severity: global_targets[severity] - used[severity]
            for severity in SEVERITY_ORDER
        }
    )
    for recipe in RECIPE_ORDER:
        slots = recipe_counts[recipe] - sum(allocations[recipe].values())
        for _ in range(slots):
            candidates = [
                severity
                for severity in SEVERITY_ORDER
                if remaining[severity] > 0
            ]
            if not candidates:
                raise AssertionError("强度均衡分配内部错误。")
            chosen = max(
                candidates,
                key=lambda severity: (
                    fractions[recipe][severity],
                    remaining[severity],
                    -SEVERITY_ORDER.index(severity),
                ),
            )
            allocations[recipe][chosen] += 1
            remaining[chosen] -= 1
    if any(remaining.values()):
        raise AssertionError("强度全局数量未能精确分配。")
    return allocations


def pair_assignments(
    gt_rows: list[dict[str, str]],
    split: str,
    mode: str,
    degradation: dict[str, Any],
) -> list[tuple[dict[str, str], str, str]]:
    fixed_seed = int(degradation.get("fixed_seed", 20260810))
    severity_weights = degradation["severity_weights"]
    if mode == "balanced-one":
        recipes = exact_recipe_assignments(len(gt_rows), degradation["recipe_weights"])
        rng = np.random.default_rng(stable_seed(fixed_seed, split, "recipe_assignment"))
        rng.shuffle(recipes)
        severities_by_index = [""] * len(gt_rows)
        allocations = balanced_severity_allocations(Counter(recipes), severity_weights)
        for recipe in RECIPE_ORDER:
            indices = [index for index, name in enumerate(recipes) if name == recipe]
            severities = [
                severity
                for severity in SEVERITY_ORDER
                for _ in range(allocations[recipe][severity])
            ]
            severity_rng = np.random.default_rng(
                stable_seed(fixed_seed, split, recipe, "severity_assignment")
            )
            severity_rng.shuffle(severities)
            for index, severity in zip(indices, severities):
                severities_by_index[index] = severity
        return list(zip(gt_rows, recipes, severities_by_index))
    if mode == "all-recipes":
        result: list[tuple[dict[str, str], str, str]] = []
        for recipe in RECIPE_ORDER:
            severities = exact_severity_assignments(len(gt_rows), severity_weights)
            severity_rng = np.random.default_rng(
                stable_seed(fixed_seed, split, recipe, "severity_assignment")
            )
            severity_rng.shuffle(severities)
            result.extend(
                (row, recipe, severity)
                for row, severity in zip(gt_rows, severities)
            )
        return result
    raise RuntimeError(f"未知生成模式：{mode}")


def clean_gt_manifest(config: dict[str, Any], split: str) -> Path:
    return config["_dataset_output_path"] / "manifests" / f"{split}_gt.csv"


def pair_manifest(config: dict[str, Any], split: str) -> Path:
    return config["_dataset_output_path"] / "manifests" / f"{split}_pairs.csv"


def select_pair_gt_rows(
    config: dict[str, Any],
    split: str,
    gt_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Select a deterministic fixed-pair subset without changing the GT snapshot."""
    source_count = len(gt_rows)
    targets = config.get("fixed_pair_target_counts", {})
    raw_target = targets.get(split)
    target = source_count if raw_target is None else int(raw_target)
    if target <= 0:
        raise RuntimeError(f"{split} 固定配对目标必须大于0，当前={target}。")
    if target > source_count:
        raise RuntimeError(
            f"{split} 固定配对需要{target}张GT，但清单中只有{source_count}张。"
        )

    base_seed = int(config.get("selection_seed", 20260810))
    subset_seed = stable_seed(base_seed, split, "fixed_pair_subset")
    if target == source_count:
        selected = list(gt_rows)
    else:
        rng = np.random.default_rng(subset_seed)
        selected_indices = sorted(
            int(index) for index in rng.choice(source_count, size=target, replace=False)
        )
        selected = [gt_rows[index] for index in selected_indices]

    selection_text = "\n".join(row.get("record_key", "") for row in selected)
    selection_hash = hashlib.sha256(selection_text.encode("utf-8")).hexdigest()
    metadata = {
        "split": split,
        "source_gt_count": source_count,
        "selected_gt_count": len(selected),
        "configured_target": raw_target,
        "pair_selection_seed": subset_seed,
        "pair_selection_sha256": selection_hash,
    }
    return selected, metadata


def make_pair_id(row: dict[str, str], recipe: str, severity: str) -> str:
    image_id = safe_component(row.get("image_id", ""), "image")
    digest = hashlib.sha256(row.get("record_key", "").encode("utf-8")).hexdigest()[:8]
    return f"{image_id}_{digest}__{recipe}__{severity}"


def pair_row_from_params(
    index: int,
    pair_id: str,
    split: str,
    gt_row: dict[str, str],
    recipe: str,
    severity: str,
    input_path: Path,
    rng_seed: int,
    params: dict[str, Any],
    mean_change: float,
    config_hash: str,
    gt_manifest_hash: str,
    mode: str,
    selection_metadata: dict[str, Any],
) -> dict[str, Any]:
    airlight = params.get("airlight", ["", "", ""])
    return {
        "pair_index": index,
        "pair_id": pair_id,
        "split": split,
        "record_key": gt_row.get("record_key", ""),
        "image_id": gt_row.get("image_id", ""),
        "recipe": recipe,
        "severity": severity,
        "haze_level": params.get("haze_level", ""),
        "dark_level": params.get("dark_level", ""),
        "noise_level": params.get("noise_level", ""),
        "has_haze": int(bool(params.get("has_haze"))),
        "has_dark": int(bool(params.get("has_dark"))),
        "has_noise": int(bool(params.get("has_noise"))),
        "operation_order": "haze>dark>noise",
        "gt_path": gt_row.get("gt_path", ""),
        "input_path": str(input_path),
        "rng_seed": rng_seed,
        "haze_airlight_r": airlight[0],
        "haze_airlight_g": airlight[1],
        "haze_airlight_b": airlight[2],
        "haze_transmission_low": params.get("transmission_low", ""),
        "haze_transmission_high": params.get("transmission_high", ""),
        "haze_grid_h": params.get("grid_h", ""),
        "haze_grid_w": params.get("grid_w", ""),
        "dark_model": params.get("dark_model", ""),
        "dark_gamma": params.get("gamma", ""),
        "dark_target_luma_p50": params.get("target_luma_p50", ""),
        "dark_source_luma_p50": params.get("source_luma_p50", ""),
        "dark_max_source_p50_ratio": params.get("max_source_p50_ratio", ""),
        "dark_effective_target_luma_p50": params.get(
            "effective_target_luma_p50", ""
        ),
        "dark_highlight_cap": params.get("highlight_cap", ""),
        "dark_scale": params.get("dark_scale", ""),
        "noise_distribution": params.get("noise_distribution", ""),
        "noise_mean_255": params.get("noise_mean_255", ""),
        "noise_sigma_255": params.get("noise_sigma_255", ""),
        "noise_gaussian_sigma": params.get("gaussian_sigma", ""),
        "noise_truncate_sigma": params.get("noise_truncate_sigma", ""),
        "mean_absolute_change": f"{mean_change:.8f}",
        "pair_source_gt_count": selection_metadata["source_gt_count"],
        "pair_selected_gt_count": selection_metadata["selected_gt_count"],
        "pair_selection_seed": selection_metadata["pair_selection_seed"],
        "pair_selection_sha256": selection_metadata["pair_selection_sha256"],
        "degradation_config_sha256": config_hash,
        "gt_manifest_sha256": gt_manifest_hash,
        "generation_mode": mode,
        "app_version": APP_VERSION,
    }


def command_generate(config: dict[str, Any], args: argparse.Namespace) -> None:
    split = args.split
    gt_path = clean_gt_manifest(config, split)
    if not gt_path.is_file():
        raise FileNotFoundError(f"找不到干净GT清单：{gt_path}；请先运行prepare。")
    _, gt_rows = read_csv(gt_path)
    if not gt_rows:
        raise RuntimeError(f"{split} GT清单为空，无法生成退化。")

    gt_rows, selection_metadata = select_pair_gt_rows(config, split, gt_rows)
    selection_metadata_path = (
        config["_dataset_output_path"]
        / "manifests"
        / f"{split}_pair_selection.json"
    )
    reproducibility_metadata = {
        **selection_metadata,
        "source_gt_manifest": str(gt_path),
        "source_gt_manifest_sha256": sha256_file(gt_path),
        "selection_seed_base": int(config.get("selection_seed", 20260810)),
        "app_version": APP_VERSION,
    }
    if selection_metadata_path.is_file():
        with selection_metadata_path.open("r", encoding="utf-8-sig") as handle:
            old_metadata = json.load(handle)
        keys_to_check = (
            "split",
            "source_gt_count",
            "selected_gt_count",
            "configured_target",
            "pair_selection_seed",
            "pair_selection_sha256",
            "source_gt_manifest_sha256",
        )
        if any(old_metadata.get(key) != reproducibility_metadata.get(key) for key in keys_to_check):
            raise RuntimeError(
                f"固定配对子集配置与已有记录不一致：{selection_metadata_path}。"
                "为保护可复现性，请勿修改已生成数据的抽样目标或种子。"
            )
    else:
        write_json_atomic(selection_metadata_path, reproducibility_metadata)

    print(
        f"{split} 固定配对子集：GT总数={selection_metadata['source_gt_count']}；"
        f"入选={selection_metadata['selected_gt_count']}；"
        f"子集SHA256={selection_metadata['pair_selection_sha256'][:12]}..."
    )

    degradation = config["degradation"]
    config_hash = json_sha256(degradation)
    gt_manifest_hash = sha256_file(gt_path)
    assignments = pair_assignments(gt_rows, split, args.mode, degradation)
    image_format = str(degradation.get("image_format", "png")).lower()
    suffix = ".png" if image_format == "png" else ".jpg"
    input_dir = config["_dataset_output_path"] / "fixed_pairs" / split / "input"
    manifest_path = pair_manifest(config, split)

    existing: dict[str, dict[str, str]] = {}
    if manifest_path.is_file():
        _, existing_rows = read_csv(manifest_path)
        for row in existing_rows:
            pair_id = row.get("pair_id", "")
            if row.get("degradation_config_sha256") != config_hash:
                raise RuntimeError(
                    f"{manifest_path} 使用了不同的退化配置。为保护测试集固定性，"
                    "请把 dataset_output 改成新版本目录后重新生成。"
                )
            if row.get("gt_manifest_sha256") != gt_manifest_hash:
                raise RuntimeError(
                    f"{manifest_path} 对应的GT清单已变化。请使用新的dataset_output版本。"
                )
            if row.get("generation_mode") != args.mode:
                raise RuntimeError(
                    f"已有生成模式={row.get('generation_mode')}，本次={args.mode}。"
                    "请勿在同一版本混用。"
                )
            existing[pair_id] = row

    ordered_rows: list[dict[str, Any]] = []
    fixed_seed = int(degradation.get("fixed_seed", 20260810))
    for index, (gt_row, recipe, severity) in enumerate(assignments, start=1):
        pair_id = make_pair_id(gt_row, recipe, severity)
        input_path = input_dir / f"{pair_id}{suffix}"
        old = existing.get(pair_id)
        if old and Path(old.get("input_path", "")).is_file():
            ordered_rows.append(old)
            continue

        seed_value = stable_seed(
            fixed_seed,
            split,
            gt_row.get("record_key", ""),
            recipe,
            severity,
            args.mode,
        )
        rng = np.random.default_rng(seed_value)
        params = sample_degradation_parameters(
            rng,
            recipe,
            severity,
            degradation,
        )
        clean = load_rgb_float(Path(gt_row["gt_path"]))
        degraded = apply_degradation(clean, params, rng)
        mean_change = float(np.mean(np.abs(degraded - clean)))
        save_rgb_float(input_path, degraded, image_format)
        item = pair_row_from_params(
            index,
            pair_id,
            split,
            gt_row,
            recipe,
            severity,
            input_path,
            seed_value,
            params,
            mean_change,
            config_hash,
            gt_manifest_hash,
            args.mode,
            selection_metadata,
        )
        ordered_rows.append(item)
        if index % 25 == 0 or index == len(assignments):
            write_csv_atomic(manifest_path, PAIR_MANIFEST_FIELDS, ordered_rows)
            print(f"{split}: {index}/{len(assignments)}")

    write_csv_atomic(manifest_path, PAIR_MANIFEST_FIELDS, ordered_rows)
    recipe_counts = Counter(row["recipe"] for row in ordered_rows)
    severity_counts = Counter(row["severity"] for row in ordered_rows)
    print(f"固定配对数据已生成：{manifest_path}")
    print("退化分布：" + "；".join(f"{k}={recipe_counts[k]}" for k in RECIPE_ORDER))
    print(
        "强度分布："
        + "；".join(f"{k}={severity_counts[k]}" for k in SEVERITY_ORDER)
    )


def thumbnail(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    result = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    left = (size[0] - result.width) // 2
    top = (size[1] - result.height) // 2
    canvas.paste(result, (left, top))
    return canvas


def command_preview(config: dict[str, Any], args: argparse.Namespace) -> None:
    split = args.split
    gt_path = clean_gt_manifest(config, split)
    if not gt_path.is_file():
        raise FileNotFoundError(f"找不到{split} GT清单；请先运行prepare。")
    _, rows = read_csv(gt_path)
    if not rows:
        raise RuntimeError(f"{split} GT清单为空。")

    degradation = config["degradation"]
    fixed_seed = int(degradation.get("fixed_seed", 20260810))
    ordered = sorted(
        rows,
        key=lambda row: stable_score(fixed_seed, row.get("record_key", "")),
    )
    chosen = ordered[0]
    clean_array = load_rgb_float(Path(chosen["gt_path"]))
    clean_image = Image.fromarray(
        np.clip(np.rint(clean_array * 255.0), 0, 255).astype(np.uint8),
        "RGB",
    )
    thumb_size = (300, 240)
    margin = 18
    label_height = 30
    columns = ("GT", "LIGHT", "MEDIUM", "HEAVY")
    width = margin * (len(columns) + 1) + thumb_size[0] * len(columns)
    row_height = thumb_size[1] + label_height + margin
    height = margin + row_height * len(RECIPE_ORDER)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for column_index, label in enumerate(columns):
        x = margin + column_index * (thumb_size[0] + margin)
        draw.text((x, 2), label, fill="black")

    for row_index, recipe in enumerate(RECIPE_ORDER):
        y = margin + row_index * row_height + label_height
        draw.text((margin, y - label_height + 6), recipe, fill="black")
        sheet.paste(thumbnail(clean_image, thumb_size), (margin, y))
        for column_index, severity in enumerate(SEVERITY_ORDER, start=1):
            seed_value = stable_seed(
                fixed_seed,
                "preview",
                split,
                chosen.get("record_key", ""),
                recipe,
                severity,
            )
            rng = np.random.default_rng(seed_value)
            params = sample_degradation_parameters(
                rng,
                recipe,
                severity,
                degradation,
            )
            degraded_array = apply_degradation(clean_array, params, rng)
            degraded_image = Image.fromarray(
                np.clip(np.rint(degraded_array * 255.0), 0, 255).astype(np.uint8),
                "RGB",
            )
            x = margin + column_index * (thumb_size[0] + margin)
            sheet.paste(thumbnail(degraded_image, thumb_size), (x, y))

    report_dir = config["_dataset_output_path"] / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / f"{split}_degradation_preview_grid.png"
    sheet.save(output, format="PNG")
    print(f"退化预览已生成：{output}")
    print("同一张GT按七类退化和轻/中/重三级展示；确认后再固定测试INPUT。")


def command_pair_preview(config: dict[str, Any], args: argparse.Namespace) -> None:
    split = args.split
    manifest_path = pair_manifest(config, split)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"找不到{split}固定配对清单：{manifest_path}；请先运行generate。"
        )
    _, rows = read_csv(manifest_path)
    if not rows:
        raise RuntimeError(f"{split}固定配对清单为空。")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("recipe", ""), row.get("severity", ""))].append(row)
    for group_rows in grouped.values():
        group_rows.sort(key=lambda row: row.get("pair_id", ""))

    thumb_size = (240, 192)
    margin = 16
    label_height = 28
    columns = tuple(
        label
        for severity in SEVERITY_ORDER
        for label in (f"{severity.upper()} GT", f"{severity.upper()} INPUT")
    )
    width = margin * (len(columns) + 1) + thumb_size[0] * len(columns)
    row_height = thumb_size[1] + label_height + margin
    height = margin + label_height + row_height * len(RECIPE_ORDER)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for column_index, label in enumerate(columns):
        x = margin + column_index * (thumb_size[0] + margin)
        draw.text((x, 4), label, fill="black")

    for row_index, recipe in enumerate(RECIPE_ORDER):
        y = margin + label_height + row_index * row_height + label_height
        draw.text((margin, y - label_height + 6), recipe, fill="black")
        for severity_index, severity in enumerate(SEVERITY_ORDER):
            candidates = grouped.get((recipe, severity), [])
            if not candidates:
                raise RuntimeError(
                    f"{split}配对中缺少{recipe}/{severity}，无法生成完整预览。"
                )
            row = candidates[0]
            with Image.open(Path(row["gt_path"])) as gt_image:
                gt_thumb = thumbnail(gt_image, thumb_size)
            with Image.open(Path(row["input_path"])) as input_image:
                input_thumb = thumbnail(input_image, thumb_size)
            gt_column = severity_index * 2
            input_column = gt_column + 1
            gt_x = margin + gt_column * (thumb_size[0] + margin)
            input_x = margin + input_column * (thumb_size[0] + margin)
            sheet.paste(gt_thumb, (gt_x, y))
            sheet.paste(input_thumb, (input_x, y))

    report_dir = config["_dataset_output_path"] / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / f"{split}_fixed_pairs_preview_grid.png"
    sheet.save(output, format="PNG")
    print(f"固定配对预览已生成：{output}")
    print("每行为一种退化；每个强度依次展示实际GT和对应INPUT。")


def verify_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        size = image.size
        image.verify()
    return size


def command_verify(config: dict[str, Any], args: argparse.Namespace) -> None:
    output_root: Path = config["_dataset_output_path"]
    seen_records: set[str] = set()
    seen_hashes: dict[str, str] = {}
    gt_lookup: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    clean_counts: dict[str, int] = {}
    pair_counts: dict[str, int] = {}

    for split in SPLITS:
        path = clean_gt_manifest(config, split)
        if not path.is_file():
            errors.append(f"缺少GT清单：{path}")
            continue
        _, rows = read_csv(path)
        clean_counts[split] = len(rows)
        for row in rows:
            key = row.get("record_key", "")
            if key in seen_records:
                errors.append(f"跨分区重复record_key：{key}")
            seen_records.add(key)
            file_hash = row.get("file_sha256", "").strip().lower()
            if file_hash:
                if file_hash in seen_hashes:
                    errors.append(
                        f"跨分区重复SHA256：{row.get('gt_path')} 与 {seen_hashes[file_hash]}"
                    )
                seen_hashes[file_hash] = row.get("gt_path", "")
            gt_file = Path(row.get("gt_path", ""))
            if not gt_file.is_file():
                errors.append(f"GT不存在：{gt_file}")
            elif not args.quick:
                try:
                    gt_size = verify_image(gt_file)
                    expected_size = (
                        int(row.get("gt_width", "0") or 0),
                        int(row.get("gt_height", "0") or 0),
                    )
                    if expected_size != (0, 0) and gt_size != expected_size:
                        errors.append(
                            f"GT尺寸与清单不一致：{gt_file}；"
                            f"实际={gt_size}；清单={expected_size}"
                        )
                    expected_hash = row.get("gt_file_sha256", "").strip().lower()
                    if expected_hash and sha256_file(gt_file).lower() != expected_hash:
                        errors.append(f"GT哈希与清单不一致：{gt_file}")
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"GT解码失败：{gt_file}；{exc}")
            gt_lookup[key] = row

    for split in SPLITS:
        path = pair_manifest(config, split)
        if not path.is_file():
            pair_counts[split] = 0
            continue
        _, rows = read_csv(path)
        pair_counts[split] = len(rows)
        seen_pair_ids: set[str] = set()
        for row in rows:
            pair_id = row.get("pair_id", "")
            if pair_id in seen_pair_ids:
                errors.append(f"重复pair_id：{pair_id}")
            seen_pair_ids.add(pair_id)
            key = row.get("record_key", "")
            if key not in gt_lookup:
                errors.append(f"配对记录找不到GT：{pair_id} / {key}")
            elif gt_lookup[key].get("split") != split:
                errors.append(f"配对跨分区：{pair_id}")
            input_file = Path(row.get("input_path", ""))
            gt_file = Path(row.get("gt_path", ""))
            if not input_file.is_file():
                errors.append(f"INPUT不存在：{input_file}")
                continue
            if not args.quick:
                try:
                    input_size = verify_image(input_file)
                    gt_size = verify_image(gt_file)
                    if input_size != gt_size:
                        errors.append(
                            f"尺寸不一致：{pair_id}；INPUT={input_size}；GT={gt_size}"
                        )
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"配对解码失败：{pair_id}；{exc}")

    print("========== 数据集核验 ==========")
    print("GT：" + "；".join(f"{k}={clean_counts.get(k, 0)}" for k in SPLITS))
    print("PAIR：" + "；".join(f"{k}={pair_counts.get(k, 0)}" for k in SPLITS))
    if errors:
        print(f"FAILED：发现{len(errors)}个问题")
        for error in errors[:50]:
            print(f"  - {error}")
        if len(errors) > 50:
            print(f"  ...另有{len(errors) - 50}项")
        raise SystemExit(2)
    print("PASSED：未发现跨分区泄漏、缺失文件、重复记录或尺寸错误。")


def mse_metric(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.mean(np.square(reference.astype(np.float64) - estimate.astype(np.float64))))


def psnr_metric(reference: np.ndarray, estimate: np.ndarray) -> float:
    mse = mse_metric(reference, estimate)
    return float("inf") if mse <= 0 else 10.0 * math.log10(1.0 / mse)


def snr_metric(reference: np.ndarray, estimate: np.ndarray) -> float:
    signal = float(np.mean(np.square(reference.astype(np.float64))))
    noise = mse_metric(reference, estimate)
    if noise <= 0:
        return float("inf")
    if signal <= 0:
        return float("-inf")
    return 10.0 * math.log10(signal / noise)


def ssim_metric(reference: np.ndarray, estimate: np.ndarray) -> float:
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as exc:
        raise RuntimeError(
            "计算SSIM需要SciPy。请运行：python -m pip install -r requirements.txt"
        ) from exc
    c1 = 0.01**2
    c2 = 0.03**2
    values: list[float] = []
    for channel in range(3):
        x = reference[..., channel].astype(np.float64)
        y = estimate[..., channel].astype(np.float64)
        mu_x = gaussian_filter(x, sigma=1.5, mode="reflect", truncate=3.5)
        mu_y = gaussian_filter(y, sigma=1.5, mode="reflect", truncate=3.5)
        sigma_x = gaussian_filter(x * x, sigma=1.5, mode="reflect", truncate=3.5) - mu_x * mu_x
        sigma_y = gaussian_filter(y * y, sigma=1.5, mode="reflect", truncate=3.5) - mu_y * mu_y
        sigma_xy = gaussian_filter(x * y, sigma=1.5, mode="reflect", truncate=3.5) - mu_x * mu_y
        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        values.append(float(np.mean(numerator / np.maximum(denominator, 1e-12))))
    return float(np.mean(values))


def build_prediction_map(directory: Path) -> dict[str, Path]:
    allowed = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    result: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        if path.stem in result:
            duplicates.add(path.stem)
        result[path.stem] = path
    if duplicates:
        raise RuntimeError(
            "预测目录存在同名文件：" + ", ".join(sorted(duplicates)[:20])
        )
    return result


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else float("nan")


def command_evaluate(config: dict[str, Any], args: argparse.Namespace) -> None:
    manifest_path = pair_manifest(config, args.split)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到配对清单：{manifest_path}")
    _, pairs = read_csv(manifest_path)
    if not pairs:
        raise RuntimeError("配对清单为空。")

    prediction_map: dict[str, Path] | None = None
    if args.prediction_dir:
        prediction_dir = Path(args.prediction_dir)
        if not prediction_dir.is_dir():
            raise FileNotFoundError(f"预测目录不存在：{prediction_dir}")
        prediction_map = build_prediction_map(prediction_dir)

    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs, start=1):
        pair_id = pair["pair_id"]
        gt = load_rgb_float(Path(pair["gt_path"]))
        degraded = load_rgb_float(Path(pair["input_path"]))
        if prediction_map is None:
            estimate = degraded
            estimate_path = pair["input_path"]
        else:
            estimate_path_obj = prediction_map.get(pair_id)
            if estimate_path_obj is None:
                raise RuntimeError(f"缺少预测结果：{pair_id}.*")
            estimate = load_rgb_float(estimate_path_obj)
            estimate_path = str(estimate_path_obj)
        if gt.shape != degraded.shape or gt.shape != estimate.shape:
            raise RuntimeError(f"图像尺寸不一致：{pair_id}")

        input_psnr = psnr_metric(gt, degraded)
        input_ssim = ssim_metric(gt, degraded)
        input_snr = snr_metric(gt, degraded)
        output_psnr = psnr_metric(gt, estimate)
        output_ssim = ssim_metric(gt, estimate)
        output_snr = snr_metric(gt, estimate)
        rows.append(
            {
                "pair_id": pair_id,
                "split": pair["split"],
                "recipe": pair["recipe"],
                "severity": pair.get("severity", ""),
                "gt_path": pair["gt_path"],
                "input_path": pair["input_path"],
                "estimate_path": estimate_path,
                "input_psnr_db": f"{input_psnr:.8f}",
                "input_ssim": f"{input_ssim:.8f}",
                "input_snr_db": f"{input_snr:.8f}",
                "output_psnr_db": f"{output_psnr:.8f}",
                "output_ssim": f"{output_ssim:.8f}",
                "output_snr_db": f"{output_snr:.8f}",
                "psnr_gain_db": f"{output_psnr - input_psnr:.8f}",
                "ssim_gain": f"{output_ssim - input_ssim:.8f}",
                "snr_gain_db": f"{output_snr - input_snr:.8f}",
            }
        )
        if index % 25 == 0 or index == len(pairs):
            print(f"评价：{index}/{len(pairs)}")

    report_dir = config["_dataset_output_path"] / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    mode_name = "baseline_input" if prediction_map is None else "model_predictions"
    fields = list(rows[0])
    output_csv = report_dir / f"{args.split}_{mode_name}_metrics.csv"
    write_csv_atomic(output_csv, fields, rows)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"recipe:{row['recipe']}"].append(row)
        groups[f"severity:{row['severity']}"].append(row)
        groups[f"recipe_severity:{row['recipe']}:{row['severity']}"].append(row)
        groups["OVERALL"].append(row)
    summary: dict[str, Any] = {
        "created_at_utc": now_utc(),
        "mode": mode_name,
        "split": args.split,
        "pair_count": len(rows),
        "metric_definition": {
            "PSNR": "10*log10(1/MSE), RGB range [0,1]",
            "SNR": "10*log10(mean(GT^2)/MSE)",
            "SSIM": "Gaussian-window RGB channel mean",
            "PSNR_gain": "model_output_PSNR - degraded_input_PSNR",
        },
        "groups": {},
    }
    for group_name, group_rows in groups.items():
        summary["groups"][group_name] = {
            "count": len(group_rows),
            "input_psnr_db": finite_mean(float(row["input_psnr_db"]) for row in group_rows),
            "input_ssim": finite_mean(float(row["input_ssim"]) for row in group_rows),
            "input_snr_db": finite_mean(float(row["input_snr_db"]) for row in group_rows),
            "output_psnr_db": finite_mean(float(row["output_psnr_db"]) for row in group_rows),
            "output_ssim": finite_mean(float(row["output_ssim"]) for row in group_rows),
            "output_snr_db": finite_mean(float(row["output_snr_db"]) for row in group_rows),
            "psnr_gain_db": finite_mean(float(row["psnr_gain_db"]) for row in group_rows),
        }
    output_json = report_dir / f"{args.split}_{mode_name}_summary.json"
    write_json_atomic(output_json, summary)
    overall = summary["groups"]["OVERALL"]
    print(f"逐图结果：{output_csv}")
    print(f"汇总结果：{output_json}")
    print(
        f"OVERALL PSNR：INPUT={overall['input_psnr_db']:.4f} dB；"
        f"OUTPUT={overall['output_psnr_db']:.4f} dB；"
        f"提升={overall['psnr_gain_db']:.4f} dB"
    )


def make_self_test_image(index: int, width: int = 96, height: int = 72) -> np.ndarray:
    x = np.linspace(0.1, 0.9, width, dtype=np.float32)[None, :]
    y = np.linspace(0.05, 0.8, height, dtype=np.float32)[:, None]
    red = np.broadcast_to((x + index * 0.013) % 1.0, (height, width))
    green = np.broadcast_to((y + index * 0.017) % 1.0, (height, width))
    blue = np.clip(0.55 * red + 0.45 * green, 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def command_self_test(_config: dict[str, Any], _args: argparse.Namespace) -> None:
    policy = _config["degradation"]["combination_policy"]
    if policy["double"]["heavy"] != ["heavy", "heavy"]:
        raise AssertionError("双重混合的最重档必须由两个heavy分量组成。")
    if policy["triple"]["heavy"] != ["heavy", "heavy", "heavy"]:
        raise AssertionError("三重混合的最重档必须由三个heavy分量组成。")
    with tempfile.TemporaryDirectory(prefix="dronevehicle_rgb_pipeline_v3_test_") as temp_name:
        root = Path(temp_name) / "DroneVehicle"
        result_path = root / "07_manual_review_v1" / "_tables" / "manual_review_results.csv"
        manual_rows: list[dict[str, Any]] = []
        split_counts = {"train": 21, "val": 0, "test": 21}
        for split, count in split_counts.items():
            source_dir = root / "01_raw" / split
            source_dir.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                image_id = f"{(index + 1) * 10:05d}"
                path = source_dir / f"{image_id}.jpg"
                scene = make_self_test_image(index + SPLITS.index(split) * 30)
                source = np.ones((104, 128, 3), dtype=np.float32)
                source[16:88, 16:112] = scene
                Image.fromarray(
                    np.clip(np.rint(source * 255.0), 0, 255).astype(np.uint8),
                    "RGB",
                ).save(path, format="JPEG", quality=95)
                manual_rows.append(
                    {
                        "record_key": f"DroneVehicle/{split}/RGB/{image_id}",
                        "split": split,
                        "manual_label": "day_clean",
                        "image_id": image_id,
                        "filename": path.name,
                        "absolute_path": str(path),
                        "file_sha256": sha256_file(path),
                        "analysis_roi": "16,16,112,88",
                    }
                )
        write_csv_atomic(
            result_path,
            [
                "record_key",
                "split",
                "manual_label",
                "image_id",
                "filename",
                "absolute_path",
                "file_sha256",
                "analysis_roi",
            ],
            manual_rows,
        )

        test_config = json.loads(
            json.dumps(
                {
                    key: value
                    for key, value in _config.items()
                    if not key.startswith("_")
                }
            )
        )
        test_config["root"] = str(root)
        test_config["dataset_output"] = "08_rgb_enhancement_dataset_v3_3"
        test_config["target_counts"] = {"train": None, "val": 0, "test": 21}
        test_config["fixed_pair_target_counts"] = {
            "train": 14,
            "val": None,
            "test": None,
        }
        test_config["min_image_id_gap"] = {"train": 0, "val": 0, "test": 2}
        test_config["degradation"]["haze"]["coarse_grid"] = [3, 5]
        config_path = root / "config.json"
        write_json_atomic(config_path, test_config)
        config = load_config(config_path)
        command_count(config, argparse.Namespace())
        prepare_args = argparse.Namespace(dry_run=False, self_test_internal=True)
        command_prepare(config, prepare_args)
        command_preview(config, argparse.Namespace(split="test"))
        command_generate(
            config, argparse.Namespace(split="test", mode="balanced-one")
        )
        command_generate(
            config, argparse.Namespace(split="train", mode="balanced-one")
        )
        command_pair_preview(config, argparse.Namespace(split="test"))
        command_verify(config, argparse.Namespace(quick=False))
        command_evaluate(
            config, argparse.Namespace(split="test", prediction_dir=None)
        )

        _, train_pairs = read_csv(pair_manifest(config, "train"))
        _, test_gt = read_csv(clean_gt_manifest(config, "test"))
        _, test_pairs = read_csv(pair_manifest(config, "test"))
        if len(train_pairs) != 14 or len(test_gt) != 21 or len(test_pairs) != 21:
            raise AssertionError("自测数量不正确。")
        train_selection_path = (
            config["_dataset_output_path"]
            / "manifests"
            / "train_pair_selection.json"
        )
        with train_selection_path.open("r", encoding="utf-8-sig") as handle:
            train_selection = json.load(handle)
        if (
            train_selection.get("source_gt_count") != 21
            or train_selection.get("selected_gt_count") != 14
        ):
            raise AssertionError("固定训练子集抽样自测失败。")
        if any(
            (int(row["gt_width"]), int(row["gt_height"])) != (96, 72)
            or row["roi_source"] != "analysis_roi"
            for row in test_gt
        ):
            raise AssertionError("ROI裁剪自测失败。")
        recipe_counts = Counter(row["recipe"] for row in test_pairs)
        severity_counts = Counter(row["severity"] for row in test_pairs)
        if any(recipe_counts[name] != 3 for name in RECIPE_ORDER):
            raise AssertionError("七类退化未均衡分配。")
        if any(severity_counts[name] != 7 for name in SEVERITY_ORDER):
            raise AssertionError("轻中重等级未均衡分配。")
        if any(row.get("noise_distribution") not in {"", "gaussian"} for row in test_pairs):
            raise AssertionError("出现非高斯主噪声。")
        dark_rows = [row for row in test_pairs if row.get("has_dark") == "1"]
        if not dark_rows or any(
            row.get("dark_model") != "global_monotonic_luminance"
            or not row.get("dark_target_luma_p50")
            or not row.get("dark_source_luma_p50")
            or not row.get("dark_max_source_p50_ratio")
            or not row.get("dark_effective_target_luma_p50")
            or not row.get("dark_highlight_cap")
            or not row.get("dark_scale")
            for row in dark_rows
        ):
            raise AssertionError("正式配对没有完整记录纯暗光模型参数。")

        dark_test_source = make_self_test_image(91)
        dark_test_outputs: list[np.ndarray] = []
        dark_test_p50: list[float] = []
        for severity in SEVERITY_ORDER:
            level = test_config["degradation"]["dark"]["levels"][severity]
            dark_params: dict[str, Any] = {
                "dark_model": "global_monotonic_luminance",
                "gamma": float(np.mean(level["gamma"])),
                "target_luma_p50": float(np.mean(level["target_luma_p50"])),
                "max_source_p50_ratio": float(level["max_source_p50_ratio"]),
                "highlight_cap": float(np.mean(level["highlight_cap"])),
            }
            dark_output = apply_dark(dark_test_source, dark_params)
            dark_test_outputs.append(dark_output)
            dark_test_p50.append(float(np.median(rgb_luma(dark_output))))
        expected_targets = (0.40, 0.32, 0.050)
        if any(
            abs(actual - expected) > 0.015
            for actual, expected in zip(dark_test_p50, expected_targets)
        ):
            raise AssertionError(f"纯暗光三档目标亮度偏离：{dark_test_p50}")
        if not all(
            bool(np.all(left + 1e-7 >= right))
            for left, right in zip(dark_test_outputs, dark_test_outputs[1:])
        ):
            raise AssertionError("纯暗光轻/中/重出现像素档位反转。")
        if not bool(np.all(dark_test_source + 1e-7 >= dark_test_outputs[0])):
            raise AssertionError("纯暗光退化中出现像素变亮。")

        constant = np.full((32, 32, 3), 0.5, dtype=np.float32)
        noise_rng = np.random.default_rng(12345)
        noise_params = {
            "gaussian_sigma": 30.0 / 255.0,
            "noise_mean_255": 0.0,
            "noise_truncate_sigma": 3.0,
        }
        noised = apply_noise(constant, noise_params, noise_rng)
        if float(np.max(np.abs(noised - constant))) > (90.0 / 255.0 + 1e-6):
            raise AssertionError("高斯噪声未按±3σ截断。")

        preview_path = (
            config["_dataset_output_path"]
            / "reports"
            / "test_degradation_preview_grid.png"
        )
        if not preview_path.is_file():
            raise AssertionError("三级退化预览未生成。")
        pair_preview_path = (
            config["_dataset_output_path"]
            / "reports"
            / "test_fixed_pairs_preview_grid.png"
        )
        if not pair_preview_path.is_file():
            raise AssertionError("实际固定配对预览未生成。")
    print(
        "SELF-TEST PASSED：ROI裁剪、纯高斯±3σ、纯全局暗光三档、"
        "七类均衡、固定配对、核验和指标计算均正常。"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DroneVehicle v3：ROI裁剪GT、纯暗光三档、组合退化与配对评价工具"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
        help="配置文件路径",
    )
    parser.add_argument("--root", default=None, help="覆盖配置中的DroneVehicle根目录")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("count", help="统计四键人工筛选数量")

    prepare = subparsers.add_parser("prepare", help="建立正式干净GT数据快照")
    prepare.add_argument("--dry-run", action="store_true", help="只检查数量，不写文件")

    preview = subparsers.add_parser("preview", help="生成七类×轻中重退化预览图")
    preview.add_argument("--split", choices=SPLITS, default="test")

    pair_preview = subparsers.add_parser(
        "preview-pairs", help="生成实际固定配对GT/INPUT预览图"
    )
    pair_preview.add_argument("--split", choices=SPLITS, default="train")

    generate = subparsers.add_parser("generate", help="生成固定配对退化数据")
    generate.add_argument("--split", choices=SPLITS, required=True)
    generate.add_argument(
        "--mode",
        choices=("balanced-one", "all-recipes"),
        default="balanced-one",
        help="每张GT一种均衡退化，或每张GT生成全部七种退化",
    )

    verify = subparsers.add_parser("verify", help="核验GT与配对数据完整性")
    verify.add_argument("--quick", action="store_true", help="跳过逐图解码检查")

    evaluate = subparsers.add_parser("evaluate", help="计算PSNR、SSIM与SNR")
    evaluate.add_argument("--split", choices=SPLITS, default="test")
    evaluate.add_argument(
        "--prediction-dir",
        default=None,
        help="模型输出目录；省略时只计算退化输入基线",
    )

    subparsers.add_parser("self-test", help="在临时小数据上运行完整自测")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config, args.root)
    commands = {
        "count": command_count,
        "prepare": command_prepare,
        "preview": command_preview,
        "preview-pairs": command_pair_preview,
        "generate": command_generate,
        "verify": command_verify,
        "evaluate": command_evaluate,
        "self-test": command_self_test,
    }
    try:
        commands[args.command](config, args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        eprint(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
