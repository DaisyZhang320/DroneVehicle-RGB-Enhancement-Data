#!/usr/bin/env python3
"""Build a portable heavy-only 200-pair test set with two degradations."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dronevehicle_rgb_pipeline_v3 as pipeline


VERSION = "2.0.0-heavy-only"
OUTPUT_NAME = "11_test200_haze_darknoise_paired_v1"
ACTIVE_RECIPES = ("haze", "dark_noise")
PAIR_FIELDS = [
    "pair_index",
    "filename",
    "image_id",
    "record_key",
    "recipe",
    "severity",
    "gt_relpath",
    "input_relpath",
    "rng_seed",
    "haze_level",
    "dark_level",
    "noise_level",
    "haze_airlight_r",
    "haze_airlight_g",
    "haze_airlight_b",
    "haze_transmission_low",
    "haze_transmission_high",
    "dark_gamma",
    "dark_target_luma_p50",
    "dark_effective_target_luma_p50",
    "dark_highlight_cap",
    "noise_sigma_255",
    "noise_truncate_sigma",
    "source_gt_sha256",
    "exported_gt_sha256",
    "input_sha256",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def find_test_manifest(root: Path) -> tuple[Path, str]:
    candidates = [
        (root / "08_rgb_enhancement_dataset_v3_3/manifests/test_gt.csv", "v3.3"),
        (root / "08_rgb_enhancement_dataset_v3_2/manifests/test_gt.csv", "v3.2"),
        (root / "08_rgb_enhancement_dataset_v3_1/manifests/test_gt.csv", "v3.1"),
        (root / "08_rgb_enhancement_dataset_v3/manifests/test_gt.csv", "v3"),
    ]
    for path, version in candidates:
        if path.is_file():
            return path, version
    checked = "\n".join(f"  - {path}" for path, _ in candidates)
    raise FileNotFoundError(f"找不到测试GT清单，已检查：\n{checked}")


def load_degradation_config() -> dict:
    with (HERE / "config.json").open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)
    pipeline.validate_config(config)
    return config["degradation"]


def make_assignments(rows: list[dict[str, str]], degradation: dict) -> list[tuple[dict[str, str], str, str]]:
    if len(rows) != 200:
        raise RuntimeError(f"测试GT必须恰好为200张，当前={len(rows)}张。")

    fixed_seed = int(degradation["fixed_seed"])
    recipes = ["haze"] * 100 + ["dark_noise"] * 100
    recipe_rng = np.random.default_rng(
        pipeline.stable_seed(fixed_seed, "test200-two-types", "recipes")
    )
    recipe_rng.shuffle(recipes)

    # This delivery is intentionally heavy-only.  Haze uses the heavy haze
    # parameters.  Dark-noise uses heavy dark plus heavy Gaussian noise.
    severities = ["heavy"] * len(rows)
    return list(zip(rows, recipes, severities))


def safe_image_id(row: dict[str, str], index: int) -> str:
    raw = row.get("image_id", "").strip() or f"image_{index:04d}"
    return pipeline.safe_component(raw, f"image_{index:04d}")


def preview(output_root: Path, rows: list[dict[str, object]]) -> Path:
    samples: list[dict[str, object]] = []
    for recipe in ACTIVE_RECIPES:
        match = next(row for row in rows if row["recipe"] == recipe)
        samples.append(match)

    thumb_size = (320, 256)
    margin = 18
    label_height = 30
    row_height = thumb_size[1] + label_height + margin
    width = margin * 3 + thumb_size[0] * 2
    height = margin + row_height * len(samples)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 3), "CLEAN GT", fill="black")
    draw.text((margin * 2 + thumb_size[0], 3), "DEGRADED INPUT", fill="black")

    for index, row in enumerate(samples):
        y = margin + index * row_height + label_height
        draw.text(
            (margin, y - label_height + 6),
            f"{str(row['recipe']).upper()} / HEAVY",
            fill="black",
        )
        with Image.open(output_root / str(row["gt_relpath"])) as image:
            sheet.paste(pipeline.thumbnail(image, thumb_size), (margin, y))
        with Image.open(output_root / str(row["input_relpath"])) as image:
            sheet.paste(
                pipeline.thumbnail(image, thumb_size),
                (margin * 2 + thumb_size[0], y),
            )

    report_path = output_root / "reports/test200_two_types_preview.png"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(report_path, format="PNG")
    return report_path


def verify(output_root: Path, rows: list[dict[str, object]]) -> None:
    if len(rows) != 200:
        raise RuntimeError(f"输出配对数量错误：{len(rows)}。")
    recipe_counts = Counter(str(row["recipe"]) for row in rows)
    if recipe_counts != Counter({"haze": 100, "dark_noise": 100}):
        raise RuntimeError(f"类别分布错误：{dict(recipe_counts)}")
    if any(str(row["severity"]) != "heavy" for row in rows):
        raise RuntimeError("发现非heavy样本；本版本只允许最重档。")
    filenames: set[str] = set()
    for row in rows:
        filename = str(row["filename"])
        if filename in filenames:
            raise RuntimeError(f"出现重复文件名：{filename}")
        filenames.add(filename)
        recipe = str(row["recipe"])
        expected_gt = f"{recipe}/GT/{filename}"
        expected_input = f"{recipe}/INPUT/{filename}"
        if row["gt_relpath"] != expected_gt or row["input_relpath"] != expected_input:
            raise RuntimeError(f"目录结构错误：{filename}")
        gt_path = output_root / expected_gt
        input_path = output_root / expected_input
        if not gt_path.is_file() or not input_path.is_file():
            raise RuntimeError(f"配对文件缺失：{filename}")
        with Image.open(gt_path) as gt_image, Image.open(input_path) as input_image:
            if gt_image.size != input_image.size:
                raise RuntimeError(f"配对尺寸不一致：{filename}")


def main() -> int:
    parser = argparse.ArgumentParser(description="生成200对最重雾化/最重暗光加噪声测试数据")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    manifest_path, source_version = find_test_manifest(root)
    source_rows = read_csv(manifest_path)
    degradation = load_degradation_config()
    assignments = make_assignments(source_rows, degradation)

    output_root = root / OUTPUT_NAME
    if output_root.exists():
        raise RuntimeError(
            f"输出目录已经存在：{output_root}\n"
            "请先确认并删除旧错误版本，再重新运行。"
        )
    for recipe in ACTIVE_RECIPES:
        (output_root / recipe / "GT").mkdir(parents=True, exist_ok=True)
        (output_root / recipe / "INPUT").mkdir(parents=True, exist_ok=True)

    output_rows: list[dict[str, object]] = []
    fixed_seed = int(degradation["fixed_seed"])
    class_indices = Counter()
    for index, (source_row, recipe, severity) in enumerate(assignments, start=1):
        source_gt = Path(source_row["gt_path"])
        if not source_gt.is_file():
            raise FileNotFoundError(f"GT不存在：{source_gt}")
        image_id = safe_image_id(source_row, index)
        class_indices[recipe] += 1
        filename = f"{recipe}_{class_indices[recipe]:03d}_heavy_{image_id}.png"
        gt_path = output_root / recipe / "GT" / filename
        input_path = output_root / recipe / "INPUT" / filename
        try:
            gt_path.hardlink_to(source_gt)
        except OSError:
            shutil.copy2(source_gt, gt_path)

        seed = pipeline.stable_seed(
            fixed_seed,
            "test200-two-types",
            source_row.get("record_key", ""),
            recipe,
            severity,
        )
        rng = np.random.default_rng(seed)
        params = pipeline.sample_degradation_parameters(
            rng,
            recipe,
            severity,
            degradation,
        )
        clean = pipeline.load_rgb_float(gt_path)
        degraded = pipeline.apply_degradation(clean, params, rng)
        pipeline.save_rgb_float(input_path, degraded, "png")

        airlight = params.get("airlight", ["", "", ""])
        output_rows.append(
            {
                "pair_index": index,
                "filename": filename,
                "image_id": source_row.get("image_id", ""),
                "record_key": source_row.get("record_key", ""),
                "recipe": recipe,
                "severity": severity,
                "gt_relpath": f"{recipe}/GT/{filename}",
                "input_relpath": f"{recipe}/INPUT/{filename}",
                "rng_seed": seed,
                "haze_level": params.get("haze_level", ""),
                "dark_level": params.get("dark_level", ""),
                "noise_level": params.get("noise_level", ""),
                "haze_airlight_r": airlight[0],
                "haze_airlight_g": airlight[1],
                "haze_airlight_b": airlight[2],
                "haze_transmission_low": params.get("transmission_low", ""),
                "haze_transmission_high": params.get("transmission_high", ""),
                "dark_gamma": params.get("gamma", ""),
                "dark_target_luma_p50": params.get("target_luma_p50", ""),
                "dark_effective_target_luma_p50": params.get("effective_target_luma_p50", ""),
                "dark_highlight_cap": params.get("highlight_cap", ""),
                "noise_sigma_255": params.get("noise_sigma_255", ""),
                "noise_truncate_sigma": params.get("noise_truncate_sigma", ""),
                "source_gt_sha256": pipeline.sha256_file(source_gt),
                "exported_gt_sha256": pipeline.sha256_file(gt_path),
                "input_sha256": pipeline.sha256_file(input_path),
            }
        )
        if index % 25 == 0 or index == 200:
            print(f"生成：{index}/200")

    verify(output_root, output_rows)
    manifest_out = output_root / "manifests/test_pairs.csv"
    write_csv(manifest_out, output_rows)
    report_path = preview(output_root, output_rows)

    recipe_counts = Counter(str(row["recipe"]) for row in output_rows)
    severity_counts = Counter(str(row["severity"]) for row in output_rows)
    metadata = {
        "version": VERSION,
        "source_manifest": str(manifest_path),
        "source_dataset_version": source_version,
        "pair_count": len(output_rows),
        "recipe_counts": dict(recipe_counts),
        "severity_counts": dict(severity_counts),
        "heavy_only": True,
        "portable_paths": True,
        "gt_input_same_filename": True,
        "degradation_config_sha256": pipeline.json_sha256(degradation),
    }
    with (output_root / "dataset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    readme = (
        "DroneVehicle测试集：雾化与暗光+噪声两类\n"
        "========================================\n"
        "样本数：200对；每张GT只对应一个INPUT。\n"
        "类别：haze=100，dark_noise=100。\n"
        "强度：全部为heavy，不包含light或medium。\n"
        "dark_noise的heavy由heavy暗光与heavy高斯噪声组合。\n"
        "同一类别下GT与INPUT使用完全相同的文件名。\n"
        "路径：haze/GT、haze/INPUT、dark_noise/GT、dark_noise/INPUT。\n"
        "本数据仅用于测试，不包含训练集。\n"
    )
    (output_root / "README.txt").write_text(readme, encoding="utf-8-sig")

    print("========== 完成 ==========")
    print(f"输出目录：{output_root}")
    print(f"配对数量：{len(output_rows)}")
    print(f"类别分布：haze={recipe_counts['haze']}；dark_noise={recipe_counts['dark_noise']}")
    print(f"强度分布：heavy={severity_counts['heavy']}；light=0；medium=0")
    print(f"清单：{manifest_out}")
    print(f"预览：{report_path}")
    print("PASSED：数量、文件、尺寸和一一配对关系均通过检查。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
