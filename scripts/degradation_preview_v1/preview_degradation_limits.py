#!/usr/bin/env python3
"""Generate a five-GT degradation-limit screening set without training.

The program reads only five deterministic rows from the approved 3967-row
training GT manifest.  It never opens the fixed 200-pair test set and never
modifies source images.  Haze is screened independently.  Dark and Gaussian
noise are screened as a complete 3 x 3 factorial design, with dark-only and
noise-only controls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw


TOOL_DIR = Path(__file__).resolve().parent
SOURCE_DIR = TOOL_DIR / "final_degradation"
sys.path.insert(0, str(SOURCE_DIR))
import dronevehicle_rgb_pipeline_v3 as pipeline


TOOL_VERSION = "1.0.0"
EXPECTED_MANIFEST_SHA256 = (
    "584622eb3ec01f998fb7c00cdaca7c365e14a1c816122e67a4e3781c9b3d0c31"
)
EXPECTED_PIPELINE_SHA256 = (
    "258fa968a1b5cbb62321268b5eb4261a658f874dc360b30843aa07ea86f0915f"
)
EXPECTED_BASE_CONFIG_SHA256 = (
    "81f3f5951b450defe6d19ebb4fea021540e4426b4be844a6726793e7eb27a642"
)
OUTPUT_NAME = "13_degradation_limit_preview_v1"
METRIC_FIELDS = [
    "image_id",
    "family",
    "candidate",
    "dark_level",
    "noise_level",
    "haze_level",
    "psnr_db",
    "ssim_luma_11x11",
    "luma_p50",
    "luma_p90",
    "dark_pixel_ratio_luma_le_16",
    "zero_quantization_ratio_channels",
    "high_quantization_ratio_channels",
    "gradient_correlation_with_gt",
    "transmission_low",
    "transmission_high",
    "airlight_r",
    "airlight_g",
    "airlight_b",
    "target_luma_p50",
    "gamma",
    "highlight_cap",
    "noise_sigma_255",
    "rng_design",
    "relative_path",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(*parts: object) -> bytes:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).digest()


def stable_seed(*parts: object) -> int:
    return int.from_bytes(stable_digest(*parts)[:8], "big") & 0x7FFFFFFF


def stable_unit(*parts: object) -> float:
    integer = int.from_bytes(stable_digest(*parts)[:8], "big")
    return integer / float((1 << 64) - 1)


def interpolate(bounds: list[float], unit: float) -> float:
    return float(bounds[0]) + (float(bounds[1]) - float(bounds[0])) * unit


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def save_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    Image.fromarray(encoded, mode="RGB").save(temporary, format="PNG", optimize=False)
    os.replace(temporary, path)


def load_policy() -> dict[str, Any]:
    policy = read_json(TOOL_DIR / "candidate_policy.json")
    if policy.get("tool_version") != TOOL_VERSION:
        raise RuntimeError("Candidate policy version mismatch.")
    if list(policy["haze"]) != ["H1", "H2", "H3"]:
        raise RuntimeError("Haze candidate order is invalid.")
    if list(policy["dark"]) != ["D1", "D2", "D3"]:
        raise RuntimeError("Dark candidate order is invalid.")
    if list(policy["noise"]) != ["N1", "N2", "N3"]:
        raise RuntimeError("Noise candidate order is invalid.")
    if policy["noise"]["N1"]["sigma_255"] != [5.0, 15.0]:
        raise RuntimeError("N1 must be the meeting-note range 5-15.")
    if policy["noise"]["N2"]["sigma_255"] != [15.0, 35.0]:
        raise RuntimeError("N2 must be the meeting-note range 15-35.")
    if policy["noise"]["N3"]["sigma_255"] != [45.0, 55.0]:
        raise RuntimeError("N3 must preserve the current extreme reference 45-55.")
    if policy["dark"]["D3"]["target_luma_p50"] != [0.045, 0.055]:
        raise RuntimeError("D3 must preserve the current extreme dark reference.")
    if policy["haze"]["H3"]["transmission_low"] != [0.12, 0.22]:
        raise RuntimeError("H3 must preserve the current extreme haze reference.")
    return policy


def validate_sources() -> None:
    if sha256_file(SOURCE_DIR / "dronevehicle_rgb_pipeline_v3.py") != EXPECTED_PIPELINE_SHA256:
        raise RuntimeError("Approved degradation implementation hash mismatch.")
    if sha256_file(SOURCE_DIR / "config.json") != EXPECTED_BASE_CONFIG_SHA256:
        raise RuntimeError("Approved base degradation config hash mismatch.")


def choose_five_rows(manifest: Path, policy: dict[str, Any]) -> list[dict[str, str]]:
    if not manifest.is_file():
        raise FileNotFoundError(f"Training GT manifest not found: {manifest}")
    actual_hash = sha256_file(manifest)
    if actual_hash != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"Training manifest hash mismatch. Expected {EXPECTED_MANIFEST_SHA256}, got {actual_hash}."
        )
    rows = read_csv(manifest)
    if len(rows) != 3967:
        raise RuntimeError(f"Expected 3967 training GT records, got {len(rows)}.")
    rows.sort(key=lambda row: int(row["image_id"]))
    boundaries = np.linspace(0, len(rows), 6, dtype=int)
    selected: list[dict[str, str]] = []
    for bin_index in range(5):
        candidates = rows[boundaries[bin_index] : boundaries[bin_index + 1]]
        chosen = min(
            candidates,
            key=lambda row: stable_digest(
                policy["selection_seed"], "five-bin-selection", bin_index, row["record_key"]
            ),
        )
        selected.append(chosen)
    if len({row["record_key"] for row in selected}) != 5:
        raise RuntimeError("Five-image selection contains duplicates.")
    return selected


def luma(image: np.ndarray) -> np.ndarray:
    return pipeline.rgb_luma(image).astype(np.float64)


def box_mean_valid(image: np.ndarray, size: int) -> np.ndarray:
    integral = np.pad(image, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    total = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return total / float(size * size)


def ssim_luma_11x11(reference: np.ndarray, candidate: np.ndarray) -> float:
    x = luma(reference)
    y = luma(candidate)
    window = 11
    ux = box_mean_valid(x, window)
    uy = box_mean_valid(y, window)
    ex2 = box_mean_valid(x * x, window)
    ey2 = box_mean_valid(y * y, window)
    exy = box_mean_valid(x * y, window)
    vx = np.maximum(ex2 - ux * ux, 0.0)
    vy = np.maximum(ey2 - uy * uy, 0.0)
    covariance = exy - ux * uy
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * ux * uy + c1) * (2.0 * covariance + c2)
    denominator = (ux * ux + uy * uy + c1) * (vx + vy + c2)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = float(np.mean((reference.astype(np.float64) - candidate.astype(np.float64)) ** 2))
    return float("inf") if mse <= 0.0 else 10.0 * math.log10(1.0 / mse)


def gradient_correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref_y = luma(reference)
    out_y = luma(candidate)
    ref_gy, ref_gx = np.gradient(ref_y)
    out_gy, out_gx = np.gradient(out_y)
    ref_magnitude = np.hypot(ref_gx, ref_gy).ravel()
    out_magnitude = np.hypot(out_gx, out_gy).ravel()
    ref_centered = ref_magnitude - float(ref_magnitude.mean())
    out_centered = out_magnitude - float(out_magnitude.mean())
    denominator = math.sqrt(
        float(np.dot(ref_centered, ref_centered))
        * float(np.dot(out_centered, out_centered))
    )
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(ref_centered, out_centered) / denominator)


def metric_values(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    y = luma(candidate)
    return {
        "psnr_db": psnr(reference, candidate),
        "ssim_luma_11x11": ssim_luma_11x11(reference, candidate),
        "luma_p50": float(np.median(y)),
        "luma_p90": float(np.percentile(y, 90.0)),
        "dark_pixel_ratio_luma_le_16": float(np.mean(y <= (16.0 / 255.0))),
        "zero_quantization_ratio_channels": float(np.mean(candidate < (0.5 / 255.0))),
        "high_quantization_ratio_channels": float(np.mean(candidate >= (254.5 / 255.0))),
        "gradient_correlation_with_gt": gradient_correlation(reference, candidate),
    }


def base_units(image_id: str) -> dict[str, float]:
    return {
        "haze_low": stable_unit(TOOL_VERSION, image_id, "haze-low"),
        "haze_high": stable_unit(TOOL_VERSION, image_id, "haze-high"),
        "airlight": stable_unit(TOOL_VERSION, image_id, "airlight"),
        "dark_target": stable_unit(TOOL_VERSION, image_id, "dark-target"),
        "dark_gamma": stable_unit(TOOL_VERSION, image_id, "dark-gamma"),
        "dark_cap": stable_unit(TOOL_VERSION, image_id, "dark-cap"),
        "noise_sigma": stable_unit(TOOL_VERSION, image_id, "noise-sigma"),
    }


def haze_candidate(
    clean: np.ndarray,
    image_id: str,
    level_name: str,
    policy: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    level = policy["haze"][level_name]
    units = base_units(image_id)
    air_bounds = policy["fixed_parameters"]["airlight"]
    air_base = interpolate(air_bounds, units["airlight"])
    jitter_rng = np.random.default_rng(stable_seed(TOOL_VERSION, image_id, "air-jitter"))
    airlight = np.clip(air_base + jitter_rng.uniform(-0.025, 0.025, 3), 0.80, 1.0)
    t_low = interpolate(level["transmission_low"], units["haze_low"])
    t_high = interpolate(level["transmission_high"], units["haze_high"])
    t_high = max(t_high, t_low + 0.05)
    grid_bounds = policy["fixed_parameters"]["coarse_grid"]
    grid_rng = np.random.default_rng(stable_seed(TOOL_VERSION, image_id, "haze-grid-size"))
    grid_h = int(grid_rng.integers(int(grid_bounds[0]), int(grid_bounds[1]) + 1))
    grid_w = int(grid_rng.integers(int(grid_bounds[0]), int(grid_bounds[1]) + 1))
    params = {
        "airlight": airlight.tolist(),
        "transmission_low": t_low,
        "transmission_high": t_high,
        "grid_h": grid_h,
        "grid_w": grid_w,
    }
    field_rng = np.random.default_rng(stable_seed(TOOL_VERSION, image_id, "shared-haze-field"))
    output = pipeline.apply_haze(clean, params, field_rng)
    return output, params


def dark_candidate(
    clean: np.ndarray,
    image_id: str,
    level_name: str,
    policy: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    level = policy["dark"][level_name]
    units = base_units(image_id)
    params = {
        "dark_model": "global_monotonic_luminance",
        "target_luma_p50": interpolate(level["target_luma_p50"], units["dark_target"]),
        "gamma": interpolate(level["gamma"], units["dark_gamma"]),
        "highlight_cap": interpolate(level["highlight_cap"], units["dark_cap"]),
        "max_source_p50_ratio": float(level["max_source_p50_ratio"]),
    }
    return pipeline.apply_dark(clean, params), params


def noise_candidate(
    image: np.ndarray,
    image_id: str,
    level_name: str,
    policy: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    level = policy["noise"][level_name]
    units = base_units(image_id)
    sigma_255 = interpolate(level["sigma_255"], units["noise_sigma"])
    params = {
        "noise_mean_255": float(policy["fixed_parameters"]["noise_mean_255"]),
        "noise_sigma_255": sigma_255,
        "gaussian_sigma": sigma_255 / 255.0,
        "noise_truncate_sigma": float(policy["fixed_parameters"]["noise_truncate_sigma"]),
    }
    noise_rng = np.random.default_rng(stable_seed(TOOL_VERSION, image_id, "shared-noise-field"))
    return pipeline.apply_noise(image, params, noise_rng), params


def make_metric_row(
    image_id: str,
    family: str,
    candidate_name: str,
    relative_path: str,
    clean: np.ndarray,
    degraded: np.ndarray,
    params: dict[str, Any],
    dark_level: str = "",
    noise_level: str = "",
    haze_level: str = "",
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "image_id": image_id,
        "family": family,
        "candidate": candidate_name,
        "dark_level": dark_level,
        "noise_level": noise_level,
        "haze_level": haze_level,
        "transmission_low": params.get("transmission_low", ""),
        "transmission_high": params.get("transmission_high", ""),
        "airlight_r": params.get("airlight", ["", "", ""])[0],
        "airlight_g": params.get("airlight", ["", "", ""])[1],
        "airlight_b": params.get("airlight", ["", "", ""])[2],
        "target_luma_p50": params.get("target_luma_p50", ""),
        "gamma": params.get("gamma", ""),
        "highlight_cap": params.get("highlight_cap", ""),
        "noise_sigma_255": params.get("noise_sigma_255", ""),
        "rng_design": "shared_quantiles_and_fields_within_each_image",
        "relative_path": relative_path.replace("\\", "/"),
    }
    values.update(metric_values(clean, degraded))
    return values


def thumbnail(array: np.ndarray, size: tuple[int, int]) -> Image.Image:
    encoded = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    image = Image.fromarray(encoded, mode="RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "black")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def save_matrix_sheet(
    path: Path,
    title: str,
    column_names: list[str],
    rows: list[tuple[str, list[np.ndarray]]],
    thumb_size: tuple[int, int] = (224, 179),
) -> None:
    margin = 14
    title_height = 34
    header_height = 30
    row_label_width = 92
    row_height = thumb_size[1] + 36
    width = row_label_width + margin * (len(column_names) + 2) + thumb_size[0] * len(column_names)
    height = title_height + header_height + margin + row_height * len(rows)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 10), title, fill="black")
    start_x = row_label_width + margin
    for column_index, name in enumerate(column_names):
        x = start_x + column_index * (thumb_size[0] + margin)
        draw.text((x, title_height + 5), name, fill="black")
    for row_index, (row_name, arrays) in enumerate(rows):
        y = title_height + header_height + margin + row_index * row_height
        draw.text((margin, y + 8), row_name, fill="black")
        for column_index, array in enumerate(arrays):
            x = start_x + column_index * (thumb_size[0] + margin)
            sheet.paste(thumbnail(array, thumb_size), (x, y + 30))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG")


def generate(root: Path, output_root: Path) -> None:
    validate_sources()
    policy = load_policy()
    manifest = root / "08_rgb_enhancement_dataset_v3_3/manifests/train_gt.csv"
    selected = choose_five_rows(manifest, policy)

    if output_root.exists():
        marker = output_root / "tool_identity.json"
        if not marker.is_file() or read_json(marker).get("tool_version") != TOOL_VERSION:
            raise RuntimeError(f"Refusing to reuse an unrelated output directory: {output_root}")
        for child_name in ("samples", "reports", "manifests"):
            child = output_root / child_name
            if child.exists():
                shutil.rmtree(child)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        output_root / "tool_identity.json",
        {
            "tool_version": TOOL_VERSION,
            "purpose": "five_image_degradation_limit_preview_only",
            "source_test_set_used": False,
            "model_training_performed": False,
        },
    )

    selection_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    cache: dict[str, dict[str, np.ndarray]] = {}

    for sample_index, row in enumerate(selected, start=1):
        gt_path = Path(row["gt_path"])
        if not gt_path.is_file():
            raise FileNotFoundError(f"Selected GT file not found: {gt_path}")
        if sha256_file(gt_path) != row["gt_file_sha256"]:
            raise RuntimeError(f"Selected GT hash mismatch: {gt_path}")
        clean = pipeline.load_rgb_float(gt_path)
        if clean.shape != (512, 640, 3):
            raise RuntimeError(f"Unexpected selected GT shape {clean.shape}: {gt_path}")

        image_id = row["image_id"]
        sample_root = output_root / "samples" / f"ID{image_id}"
        save_rgb(sample_root / "GT.png", clean)
        cache[image_id] = {"GT": clean}
        selection_rows.append(
            {
                "sample_index": sample_index,
                "image_id": image_id,
                "record_key": row["record_key"],
                "gt_path": row["gt_path"],
                "gt_file_sha256": row["gt_file_sha256"],
                "saved_gt_relative_path": f"samples/ID{image_id}/GT.png",
                "source_split": "train",
            }
        )

        for haze_level in policy["haze"]:
            degraded, params = haze_candidate(clean, image_id, haze_level, policy)
            relative = f"samples/ID{image_id}/haze/{haze_level}.png"
            save_rgb(output_root / relative, degraded)
            cache[image_id][f"haze_{haze_level}"] = degraded
            metric_rows.append(
                make_metric_row(
                    image_id,
                    "haze",
                    haze_level,
                    relative,
                    clean,
                    degraded,
                    params,
                    haze_level=haze_level,
                )
            )

        dark_outputs: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
        for dark_level in policy["dark"]:
            degraded, params = dark_candidate(clean, image_id, dark_level, policy)
            dark_outputs[dark_level] = (degraded, params)
            relative = f"samples/ID{image_id}/dark_only/{dark_level}.png"
            save_rgb(output_root / relative, degraded)
            cache[image_id][f"dark_{dark_level}"] = degraded
            metric_rows.append(
                make_metric_row(
                    image_id,
                    "dark_only_control",
                    dark_level,
                    relative,
                    clean,
                    degraded,
                    params,
                    dark_level=dark_level,
                )
            )

        for noise_level in policy["noise"]:
            degraded, params = noise_candidate(clean, image_id, noise_level, policy)
            relative = f"samples/ID{image_id}/noise_only/{noise_level}.png"
            save_rgb(output_root / relative, degraded)
            cache[image_id][f"noise_{noise_level}"] = degraded
            metric_rows.append(
                make_metric_row(
                    image_id,
                    "noise_only_control",
                    noise_level,
                    relative,
                    clean,
                    degraded,
                    params,
                    noise_level=noise_level,
                )
            )

        for dark_level in policy["dark"]:
            dark_image, dark_params = dark_outputs[dark_level]
            for noise_level in policy["noise"]:
                degraded, noise_params = noise_candidate(
                    dark_image, image_id, noise_level, policy
                )
                params = dict(dark_params)
                params.update(noise_params)
                candidate_name = f"{dark_level}_{noise_level}"
                relative = f"samples/ID{image_id}/dark_noise/{candidate_name}.png"
                save_rgb(output_root / relative, degraded)
                cache[image_id][f"dark_noise_{candidate_name}"] = degraded
                metric_rows.append(
                    make_metric_row(
                        image_id,
                        "dark_noise",
                        candidate_name,
                        relative,
                        clean,
                        degraded,
                        params,
                        dark_level=dark_level,
                        noise_level=noise_level,
                    )
                )

        print(f"Generated sample {sample_index}/5: ID{image_id}")

    write_csv(
        output_root / "manifests/selected_five_gt.csv",
        list(selection_rows[0]),
        selection_rows,
    )
    write_csv(output_root / "reports/degradation_metrics.csv", METRIC_FIELDS, metric_rows)
    write_json(output_root / "reports/candidate_policy_used.json", policy)

    ids = [row["image_id"] for row in selection_rows]
    save_matrix_sheet(
        output_root / "reports/haze_three_levels_preview.png",
        "HAZE LIMIT SCREENING - SAME GT, SHARED HAZE FIELD",
        ["CLEAN GT", "H1 LIGHTER", "H2 MODERATE", "H3 CURRENT EXTREME"],
        [
            (
                f"ID {image_id}",
                [cache[image_id]["GT"]]
                + [cache[image_id][f"haze_{level}"] for level in policy["haze"]],
            )
            for image_id in ids
        ],
    )
    save_matrix_sheet(
        output_root / "reports/dark_three_levels_preview.png",
        "DARK-ONLY CONTROL - NO NOISE",
        ["CLEAN GT", "D1 LIGHTER", "D2 MODERATE", "D3 CURRENT EXTREME"],
        [
            (
                f"ID {image_id}",
                [cache[image_id]["GT"]]
                + [cache[image_id][f"dark_{level}"] for level in policy["dark"]],
            )
            for image_id in ids
        ],
    )
    save_matrix_sheet(
        output_root / "reports/noise_three_levels_preview.png",
        "NOISE-ONLY CONTROL - NO DARKENING",
        ["CLEAN GT", "N1 SIGMA 5-15", "N2 SIGMA 15-35", "N3 SIGMA 45-55"],
        [
            (
                f"ID {image_id}",
                [cache[image_id]["GT"]]
                + [cache[image_id][f"noise_{level}"] for level in policy["noise"]],
            )
            for image_id in ids
        ],
    )
    save_matrix_sheet(
        output_root / "reports/dark_noise_key_combinations_preview.png",
        "DARK + NOISE KEY COMBINATIONS",
        ["CLEAN GT", "D1N1", "D1N2", "D2N1", "D2N2", "D3N3"],
        [
            (
                f"ID {image_id}",
                [cache[image_id]["GT"]]
                + [
                    cache[image_id][f"dark_noise_{candidate}"]
                    for candidate in ("D1_N1", "D1_N2", "D2_N1", "D2_N2", "D3_N3")
                ],
            )
            for image_id in ids
        ],
        thumb_size=(192, 154),
    )
    for image_id in ids:
        save_matrix_sheet(
            output_root / f"reports/dark_noise_3x3_ID{image_id}.png",
            f"DARK + NOISE FULL FACTORIAL - ID {image_id}",
            ["CLEAN GT", "DARK ONLY", "+ N1 5-15", "+ N2 15-35", "+ N3 45-55"],
            [
                (
                    dark_level,
                    [cache[image_id]["GT"], cache[image_id][f"dark_{dark_level}"]]
                    + [
                        cache[image_id][f"dark_noise_{dark_level}_{noise_level}"]
                        for noise_level in policy["noise"]
                    ],
                )
                for dark_level in policy["dark"]
            ],
            thumb_size=(208, 166),
        )

    write_summary(output_root, metric_rows, selection_rows)
    verify(output_root)
    print(f"Output directory: {output_root}")
    print("No model was trained; no fixed test image was opened or modified.")


def pooled_psnr(rows: list[dict[str, Any]]) -> float:
    mse_values = [10.0 ** (-float(row["psnr_db"]) / 10.0) for row in rows]
    return -10.0 * math.log10(float(np.mean(mse_values)))


def write_summary(
    output_root: Path,
    metric_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
) -> None:
    candidates: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for family in ("haze", "dark_only_control", "noise_only_control", "dark_noise"):
        family_rows = [row for row in metric_rows if row["family"] == family]
        for candidate_name in sorted({row["candidate"] for row in family_rows}):
            rows = [row for row in family_rows if row["candidate"] == candidate_name]
            summary = {
                "count": len(rows),
                "pooled_psnr_db": pooled_psnr(rows),
                "mean_ssim_luma_11x11": float(
                    np.mean([float(row["ssim_luma_11x11"]) for row in rows])
                ),
                "mean_dark_pixel_ratio_luma_le_16": float(
                    np.mean([float(row["dark_pixel_ratio_luma_le_16"]) for row in rows])
                ),
                "mean_zero_quantization_ratio_channels": float(
                    np.mean([float(row["zero_quantization_ratio_channels"]) for row in rows])
                ),
                "mean_gradient_correlation_with_gt": float(
                    np.mean([float(row["gradient_correlation_with_gt"]) for row in rows])
                ),
            }
            candidates[f"{family}/{candidate_name}"] = summary
            summary_rows.append(
                {
                    "family": family,
                    "candidate": candidate_name,
                    **summary,
                }
            )
    write_csv(
        output_root / "reports/candidate_summary.csv",
        [
            "family",
            "candidate",
            "count",
            "pooled_psnr_db",
            "mean_ssim_luma_11x11",
            "mean_dark_pixel_ratio_luma_le_16",
            "mean_zero_quantization_ratio_channels",
            "mean_gradient_correlation_with_gt",
        ],
        summary_rows,
    )
    write_json(
        output_root / "reports/degradation_summary.json",
        {
            "tool_version": TOOL_VERSION,
            "selected_train_gt_count": len(selection_rows),
            "metric_row_count": len(metric_rows),
            "test_set_used": False,
            "training_performed": False,
            "ssim_definition": (
                "luminance SSIM using an 11x11 uniform local window, "
                "C1=(0.01)^2 and C2=(0.03)^2"
            ),
            "warning": (
                "Whole-image metrics cannot prove vehicle-information retention. "
                "Vehicle-ROI and detection-label evaluation is required after screening."
            ),
            "candidates": candidates,
        },
    )


def verify(output_root: Path) -> None:
    selection_path = output_root / "manifests/selected_five_gt.csv"
    metric_path = output_root / "reports/degradation_metrics.csv"
    if not selection_path.is_file() or not metric_path.is_file():
        raise FileNotFoundError("Preview manifests are incomplete.")
    selections = read_csv(selection_path)
    metrics = read_csv(metric_path)
    if len(selections) != 5:
        raise RuntimeError(f"Expected 5 selected GT rows, got {len(selections)}.")
    if len(metrics) != 90:
        raise RuntimeError(f"Expected 90 degradation metric rows, got {len(metrics)}.")
    expected_counts = {
        "haze": 15,
        "dark_only_control": 15,
        "noise_only_control": 15,
        "dark_noise": 45,
    }
    actual_counts = Counter(row["family"] for row in metrics)
    if dict(actual_counts) != expected_counts:
        raise RuntimeError(f"Unexpected metric family counts: {dict(actual_counts)}")
    expected_candidates = {
        "haze": {"H1", "H2", "H3"},
        "dark_only_control": {"D1", "D2", "D3"},
        "noise_only_control": {"N1", "N2", "N3"},
        "dark_noise": {
            f"D{dark_index}_N{noise_index}"
            for dark_index in range(1, 4)
            for noise_index in range(1, 4)
        },
    }
    for family, candidate_names in expected_candidates.items():
        actual_names = {row["candidate"] for row in metrics if row["family"] == family}
        if actual_names != candidate_names:
            raise RuntimeError(
                f"Unexpected candidates for {family}: {sorted(actual_names)}"
            )
        for candidate_name in candidate_names:
            count = sum(
                row["family"] == family and row["candidate"] == candidate_name
                for row in metrics
            )
            if count != 5:
                raise RuntimeError(
                    f"Expected 5 rows for {family}/{candidate_name}, got {count}."
                )
    for row in selections:
        gt = output_root / row["saved_gt_relative_path"]
        if not gt.is_file():
            raise FileNotFoundError(f"Saved GT is missing: {gt}")
    for row in metrics:
        image_path = output_root / row["relative_path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Degraded image is missing: {image_path}")
        with Image.open(image_path) as image:
            if image.size != (640, 512) or image.mode != "RGB":
                raise RuntimeError(f"Unexpected image format: {image_path}")
    required_reports = [
        "haze_three_levels_preview.png",
        "dark_three_levels_preview.png",
        "noise_three_levels_preview.png",
        "dark_noise_key_combinations_preview.png",
        "candidate_summary.csv",
        "degradation_summary.json",
    ]
    for name in required_reports:
        if not (output_root / "reports" / name).is_file():
            raise FileNotFoundError(f"Required report is missing: {name}")
    print("VERIFY PASSED")
    print("Selected train GT=5; saved GT=5; degradation outputs=90.")
    print("Haze=15; dark-only controls=15; noise-only controls=15; dark-noise=45.")
    print("Dark-noise factorial coverage=5 images x 3 dark levels x 3 noise levels.")


def self_test() -> None:
    validate_sources()
    policy = load_policy()
    height, width = 128, 160
    x = np.linspace(0.05, 0.95, width, dtype=np.float32)
    y = np.linspace(0.0, 0.15, height, dtype=np.float32)[:, None]
    gray = np.clip(x[None, :] + y, 0.0, 1.0)
    clean = np.stack((gray, np.sqrt(gray), gray**1.3), axis=2).astype(np.float32)
    image_id = "SELFTEST"

    haze_psnr = []
    for level in policy["haze"]:
        output, _ = haze_candidate(clean, image_id, level, policy)
        haze_psnr.append(psnr(clean, output))
    if not (haze_psnr[0] > haze_psnr[1] > haze_psnr[2]):
        raise RuntimeError(f"Haze severity monotonicity failed: {haze_psnr}")

    dark_p50 = []
    dark_outputs = {}
    for level in policy["dark"]:
        output, _ = dark_candidate(clean, image_id, level, policy)
        dark_outputs[level] = output
        dark_p50.append(float(np.median(luma(output))))
    if not (dark_p50[0] > dark_p50[1] > dark_p50[2]):
        raise RuntimeError(f"Dark severity monotonicity failed: {dark_p50}")

    noise_mse = []
    for level in policy["noise"]:
        output, _ = noise_candidate(clean, image_id, level, policy)
        noise_mse.append(float(np.mean((output - clean) ** 2)))
    if not (noise_mse[0] < noise_mse[1] < noise_mse[2]):
        raise RuntimeError(f"Noise severity monotonicity failed: {noise_mse}")

    for dark_level, dark_image in dark_outputs.items():
        for noise_level in policy["noise"]:
            output, _ = noise_candidate(dark_image, image_id, noise_level, policy)
            if output.shape != clean.shape or not np.isfinite(output).all():
                raise RuntimeError(f"Invalid combined output: {dark_level}/{noise_level}")
    if abs(ssim_luma_11x11(clean, clean) - 1.0) > 1e-6:
        raise RuntimeError("SSIM identity self-test failed.")
    print("SELF-TEST PASSED")
    print(f"Haze PSNR H1/H2/H3={haze_psnr}")
    print(f"Dark luminance P50 D1/D2/D3={dark_p50}")
    print(f"Noise MSE N1/N2/N3={noise_mse}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("preview", "verify", "self-test"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output_root = root / OUTPUT_NAME
    if args.action == "preview":
        generate(root, output_root)
    elif args.action == "verify":
        verify(output_root)
    else:
        self_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
