#!/usr/bin/env python3
"""Controlled DroneVehicle U-Net degradation-limit pilot comparison.

The tool never copies or modifies the clean GT collection. It creates CSV/JSON
split manifests and trains one of six explicitly named degradation policies.
Every policy uses the same U-Net, split, initialization, optimizer, data order,
and random fields. Only haze/dark/noise parameter ranges differ. An 80-image,
recipe-balanced validation subset is used only for model selection.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw


TOOL_DIR = Path(__file__).resolve().parent
SOURCE_DIR = TOOL_DIR / "final_degradation"
sys.path.insert(0, str(SOURCE_DIR))
import dronevehicle_rgb_pipeline_v3 as degradation_pipeline


TOOL_VERSION = "1.1.0-limit-pilot"
EXPECTED_TRAIN_MANIFEST_SHA256 = (
    "584622eb3ec01f998fb7c00cdaca7c365e14a1c816122e67a4e3781c9b3d0c31"
)
EXPECTED_DEGRADATION_CONFIG_SHA256 = (
    "81f3f5951b450defe6d19ebb4fea021540e4426b4be844a6726793e7eb27a642"
)
EXPECTED_DEGRADATION_CODE_SHA256 = (
    "258fa968a1b5cbb62321268b5eb4261a658f874dc360b30843aa07ea86f0915f"
)
EXPECTED_REPO_COMMIT = "6e41c63"
FULL_TRAIN_COUNT = 3567
FULL_VAL_COUNT = 400
PILOT_TRAIN_COUNT = 512
PILOT_VAL_COUNT = 80
GROUP_MAX_SELECTED_ID_GAP = 20
SPLIT_SEED = 20260903
PILOT_EPOCHS = 10
TRAIN_CROP_SIZE = 256
VAL_CROP_SIZE = 256
BATCH_SIZE = 4
LEARNING_RATE = 2.0e-4
ACTIVE_RECIPES = ("haze", "dark_noise")
POLICY_NAMES = ("P0", "P1", "P2", "P3", "P4", "P5")


SPLIT_FIELDS = [
    "subset",
    "group_id",
    "group_first_image_id",
    "group_last_image_id",
    "group_size",
    "recipe",
    "severity",
    "dataset_id",
    "record_key",
    "image_id",
    "export_filename",
    "gt_path",
    "gt_file_sha256",
    "source_path",
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp, path)


def load_candidate_policy() -> dict[str, Any]:
    path = TOOL_DIR / "candidate_policy.json"
    with path.open("r", encoding="utf-8-sig") as handle:
        policy = json.load(handle)
    if tuple(policy.get("pilot_policies", {})) != POLICY_NAMES:
        raise RuntimeError("Pilot policy names or order do not match P0-P5.")
    if policy["noise"]["N1"]["sigma_255"] != [5.0, 15.0]:
        raise RuntimeError("N1 must be the meeting-note range 5-15.")
    if policy["noise"]["N2"]["sigma_255"] != [15.0, 35.0]:
        raise RuntimeError("N2 must be the meeting-note range 15-35.")
    return policy


def load_degradation_config(policy_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if policy_name not in POLICY_NAMES:
        raise RuntimeError(f"Unknown pilot policy: {policy_name}")
    config_path = SOURCE_DIR / "config.json"
    code_path = SOURCE_DIR / "dronevehicle_rgb_pipeline_v3.py"
    if sha256_file(config_path) != EXPECTED_DEGRADATION_CONFIG_SHA256:
        raise RuntimeError("The approved degradation config hash does not match.")
    if sha256_file(code_path) != EXPECTED_DEGRADATION_CODE_SHA256:
        raise RuntimeError("The approved degradation implementation hash does not match.")
    with config_path.open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)
    degradation_pipeline.validate_config(config)
    degradation = copy.deepcopy(config["degradation"])
    heavy_haze = degradation["haze"]["levels"]["heavy"]
    heavy_dark = degradation["dark"]["levels"]["heavy"]
    heavy_noise = degradation["noise"]["levels"]["heavy"]
    if heavy_haze["transmission_low"] != [0.12, 0.22]:
        raise RuntimeError("Unexpected heavy haze parameters.")
    if heavy_dark["target_luma_p50"] != [0.045, 0.055]:
        raise RuntimeError("Unexpected heavy dark parameters.")
    if heavy_noise["sigma_255"] != [45.0, 55.0]:
        raise RuntimeError("Unexpected heavy noise parameters.")
    candidates = load_candidate_policy()
    selected = candidates["pilot_policies"][policy_name]
    degradation["haze"]["levels"]["heavy"] = copy.deepcopy(
        candidates["haze"][selected["haze"]]
    )
    degradation["dark"]["levels"]["heavy"] = copy.deepcopy(
        candidates["dark"][selected["dark"]]
    )
    degradation["noise"]["levels"]["heavy"] = copy.deepcopy(
        candidates["noise"][selected["noise"]]
    )
    degradation["policy_name"] = policy_name
    degradation["policy_components"] = {
        "haze": selected["haze"],
        "dark": selected["dark"],
        "noise": selected["noise"],
        "purpose": selected["purpose"],
    }
    return degradation, degradation["policy_components"]


def write_policy_snapshot(run_dir: Path, policy_name: str) -> Path:
    candidates = load_candidate_policy()
    selected = candidates["pilot_policies"][policy_name]
    snapshot = {
        "tool_version": TOOL_VERSION,
        "policy": policy_name,
        "purpose": selected["purpose"],
        "components": {
            "haze": selected["haze"],
            "dark": selected["dark"],
            "noise": selected["noise"],
        },
        "exact_ranges": {
            "haze": candidates["haze"][selected["haze"]],
            "dark": candidates["dark"][selected["dark"]],
            "noise": candidates["noise"][selected["noise"]],
        },
        "fixed_parameters": candidates["fixed_parameters"],
        "training": {
            "model": "UNet",
            "senior_repo_commit": EXPECTED_REPO_COMMIT,
            "train_count": PILOT_TRAIN_COUNT,
            "validation_count": PILOT_VAL_COUNT,
            "epochs": PILOT_EPOCHS,
            "crop_size": TRAIN_CROP_SIZE,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "loss": "L1",
            "precision": "CUDA FP16",
            "split_and_initialization_seed": SPLIT_SEED,
        },
        "source_integrity": {
            "train_manifest_sha256": EXPECTED_TRAIN_MANIFEST_SHA256,
            "degradation_config_sha256": EXPECTED_DEGRADATION_CONFIG_SHA256,
            "degradation_code_sha256": EXPECTED_DEGRADATION_CODE_SHA256,
        },
        "fixed_test_set_used": False,
    }
    path = run_dir / "policy_config.json"
    write_json(path, snapshot)
    return path


def validate_train_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Training GT manifest not found: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != EXPECTED_TRAIN_MANIFEST_SHA256:
        raise RuntimeError(
            "Training GT manifest hash mismatch. "
            f"Expected {EXPECTED_TRAIN_MANIFEST_SHA256}, got {actual_hash}."
        )
    rows = read_csv(path)
    if len(rows) != FULL_TRAIN_COUNT + FULL_VAL_COUNT:
        raise RuntimeError(f"Expected 3967 training GT records, got {len(rows)}.")
    required = {
        "dataset_id",
        "split",
        "record_key",
        "image_id",
        "gt_path",
        "gt_file_sha256",
        "manual_label",
        "gt_width",
        "gt_height",
    }
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"Training manifest is missing fields: {sorted(missing)}")
    keys: set[str] = set()
    ids: set[int] = set()
    for row in rows:
        if row["split"] != "train" or row["manual_label"] != "day_clean":
            raise RuntimeError(f"Unexpected training GT label: {row['record_key']}")
        image_id = int(row["image_id"])
        if row["record_key"] in keys or image_id in ids:
            raise RuntimeError(f"Duplicate training GT record: {row['record_key']}")
        keys.add(row["record_key"])
        ids.add(image_id)
        if row["gt_width"] != "640" or row["gt_height"] != "512":
            raise RuntimeError(f"Unexpected GT dimensions: {row['record_key']}")
        if not Path(row["gt_path"]).is_file():
            raise FileNotFoundError(f"GT file not found: {row['gt_path']}")
    return sorted(rows, key=lambda item: int(item["image_id"]))


def build_contiguous_groups(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    groups: list[list[dict[str, str]]] = []
    current = [rows[0]]
    for row in rows[1:]:
        current_id = int(row["image_id"])
        previous_id = int(current[-1]["image_id"])
        if current_id - previous_id <= GROUP_MAX_SELECTED_ID_GAP:
            current.append(row)
        else:
            groups.append(current)
            current = [row]
    groups.append(current)
    return groups


def choose_validation_groups(groups: list[list[dict[str, str]]]) -> tuple[int, ...]:
    order = sorted(
        range(len(groups)),
        key=lambda index: stable_digest(
            SPLIT_SEED,
            groups[index][0]["image_id"],
            groups[index][-1]["image_id"],
        ),
    )
    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for index in order:
        size = len(groups[index])
        snapshot = list(reachable.items())
        for total, selected in snapshot:
            new_total = total + size
            if new_total > FULL_VAL_COUNT:
                continue
            candidate = selected + (index,)
            if new_total not in reachable or len(candidate) < len(reachable[new_total]):
                reachable[new_total] = candidate
    if FULL_VAL_COUNT not in reachable:
        raise RuntimeError("Could not create an exact 400-image group-safe validation set.")
    return tuple(sorted(reachable[FULL_VAL_COUNT]))


def to_split_row(
    row: dict[str, str],
    subset: str,
    group_id: int,
    group: list[dict[str, str]],
    recipe: str,
) -> dict[str, str]:
    return {
        "subset": subset,
        "group_id": f"G{group_id:03d}",
        "group_first_image_id": group[0]["image_id"],
        "group_last_image_id": group[-1]["image_id"],
        "group_size": str(len(group)),
        "recipe": recipe,
        "severity": "policy_controlled",
        "dataset_id": row["dataset_id"],
        "record_key": row["record_key"],
        "image_id": row["image_id"],
        "export_filename": row.get("export_filename", ""),
        "gt_path": row["gt_path"],
        "gt_file_sha256": row["gt_file_sha256"],
        "source_path": row.get("source_path", ""),
    }


def select_stable(rows: list[dict[str, str]], count: int, label: str) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: stable_digest(SPLIT_SEED, label, row["record_key"]),
    )[:count]


def prepare(root: Path, run_dir: Path) -> dict[str, Any]:
    manifest = root / "08_rgb_enhancement_dataset_v3_3/manifests/train_gt.csv"
    rows = validate_train_manifest(manifest)
    groups = build_contiguous_groups(rows)
    val_group_indices = set(choose_validation_groups(groups))

    train_rows: list[dict[str, str]] = []
    val_candidates: list[tuple[dict[str, str], int, list[dict[str, str]]]] = []
    for group_id, group in enumerate(groups, start=1):
        if group_id - 1 in val_group_indices:
            val_candidates.extend((row, group_id, group) for row in group)
        else:
            train_rows.extend(
                to_split_row(row, "train", group_id, group, "dynamic_balanced")
                for row in group
            )

    val_candidates = sorted(
        val_candidates,
        key=lambda item: stable_digest(SPLIT_SEED, "val-recipe", item[0]["record_key"]),
    )
    val_rows: list[dict[str, str]] = []
    for index, (row, group_id, group) in enumerate(val_candidates):
        recipe = "haze" if index < FULL_VAL_COUNT // 2 else "dark_noise"
        val_rows.append(to_split_row(row, "val", group_id, group, recipe))
    val_rows.sort(key=lambda item: int(item["image_id"]))

    if len(train_rows) != FULL_TRAIN_COUNT or len(val_rows) != FULL_VAL_COUNT:
        raise RuntimeError("Unexpected full train/validation counts.")
    train_keys = {row["record_key"] for row in train_rows}
    val_keys = {row["record_key"] for row in val_rows}
    if train_keys & val_keys:
        raise RuntimeError("Training/validation record leakage detected.")

    train_ids = sorted(int(row["image_id"]) for row in train_rows)
    val_ids = sorted(int(row["image_id"]) for row in val_rows)
    min_cross_gap = min(abs(train_id - val_id) for train_id in train_ids for val_id in val_ids)
    if min_cross_gap <= GROUP_MAX_SELECTED_ID_GAP:
        raise RuntimeError(f"Training/validation ID guard failed: {min_cross_gap}")

    pilot_train = select_stable(train_rows, PILOT_TRAIN_COUNT, "pilot-train")
    haze_val = select_stable(
        [row for row in val_rows if row["recipe"] == "haze"],
        PILOT_VAL_COUNT // 2,
        "pilot-val-haze",
    )
    dark_noise_val = select_stable(
        [row for row in val_rows if row["recipe"] == "dark_noise"],
        PILOT_VAL_COUNT // 2,
        "pilot-val-dark-noise",
    )
    pilot_val = sorted(haze_val + dark_noise_val, key=lambda item: int(item["image_id"]))

    manifests = run_dir / "manifests"
    write_csv(manifests / "full_train.csv", SPLIT_FIELDS, train_rows)
    write_csv(manifests / "full_val.csv", SPLIT_FIELDS, val_rows)
    write_csv(manifests / "pilot_train.csv", SPLIT_FIELDS, pilot_train)
    write_csv(manifests / "pilot_val.csv", SPLIT_FIELDS, pilot_val)

    selected_groups = []
    for index in sorted(val_group_indices):
        group = groups[index]
        selected_groups.append(
            {
                "group_id": f"G{index + 1:03d}",
                "count": len(group),
                "first_image_id": group[0]["image_id"],
                "last_image_id": group[-1]["image_id"],
            }
        )
    metadata = {
        "tool_version": TOOL_VERSION,
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256_file(manifest),
        "split_seed": SPLIT_SEED,
        "group_max_selected_id_gap": GROUP_MAX_SELECTED_ID_GAP,
        "group_count": len(groups),
        "minimum_train_val_image_id_gap": min_cross_gap,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "pilot_train_count": len(pilot_train),
        "pilot_val_count": len(pilot_val),
        "validation_recipe_counts": {"haze": 200, "dark_noise": 200},
        "validation_groups": selected_groups,
        "no_images_copied": True,
    }
    write_json(manifests / "split_metadata.json", metadata)

    print("PREPARE PASSED")
    print(f"Full split: train={len(train_rows)}, val={len(val_rows)}")
    print(f"Pilot split: train={len(pilot_train)}, val={len(pilot_val)}")
    print(f"Contiguous groups: {len(groups)}; cross-split ID gap >= {min_cross_gap}")
    for group in selected_groups:
        print(
            "VAL GROUP: "
            f"{group['group_id']} {group['first_image_id']}-{group['last_image_id']} "
            f"count={group['count']}"
        )
    print("No GT image was copied or modified.")
    return metadata


def ensure_prepared(root: Path, run_dir: Path) -> None:
    required = [
        run_dir / "manifests/full_train.csv",
        run_dir / "manifests/full_val.csv",
        run_dir / "manifests/pilot_train.csv",
        run_dir / "manifests/pilot_val.csv",
    ]
    if not all(path.is_file() for path in required):
        prepare(root, run_dir)


def make_degraded(
    clean: np.ndarray,
    row: dict[str, str],
    split: str,
    index: int,
    epoch: int,
    degradation: dict[str, Any],
) -> tuple[np.ndarray, str, int, dict[str, Any]]:
    if split == "train":
        recipe = ACTIVE_RECIPES[(index + epoch) % len(ACTIVE_RECIPES)]
        epoch_token: object = epoch
    else:
        recipe = row["recipe"]
        epoch_token = "fixed"
    seed = degradation_pipeline.stable_seed(
        degradation["fixed_seed"],
        "dronevehicle-unet-limit-pilot-v1-shared-randomness",
        split,
        row["record_key"],
        recipe,
        epoch_token,
    )
    rng = np.random.default_rng(seed)
    params = degradation_pipeline.sample_degradation_parameters(
        rng, recipe, "heavy", degradation
    )
    if recipe == "haze" and params.get("haze_level") != "heavy":
        raise RuntimeError("Haze sample did not use the heavy haze level.")
    if recipe == "dark_noise" and (
        params.get("dark_level") != "heavy" or params.get("noise_level") != "heavy"
    ):
        raise RuntimeError("Dark-noise sample did not use both heavy components.")
    degraded = degradation_pipeline.apply_degradation(clean, params, rng)
    return degraded, recipe, seed, params


def thumbnail(array: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(
        np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8), mode="RGB"
    )
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "black")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def preview(root: Path, run_dir: Path, policy_name: str) -> Path:
    ensure_prepared(root, run_dir)
    degradation, policy_components = load_degradation_config(policy_name)
    rows = read_csv(run_dir / "manifests/pilot_val.csv")
    chosen = []
    for recipe in ACTIVE_RECIPES:
        chosen.extend([row for row in rows if row["recipe"] == recipe][:4])

    thumb_size = (320, 256)
    margin = 18
    label_height = 34
    row_height = thumb_size[1] + label_height + margin
    width = margin * 3 + thumb_size[0] * 2
    height = 42 + row_height * len(chosen)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 10), "CLEAN GT", fill="black")
    draw.text(
        (margin * 2 + thumb_size[0], 10),
        f"{policy_name} INPUT: {policy_components['haze']} + "
        f"{policy_components['dark']}{policy_components['noise']}",
        fill="black",
    )

    report_rows = []
    for index, row in enumerate(chosen):
        clean = degradation_pipeline.load_rgb_float(Path(row["gt_path"]))
        degraded, recipe, seed, params = make_degraded(
            clean, row, "val", index, 0, degradation
        )
        y = 42 + index * row_height
        draw.text(
            (margin, y + 8),
            f"{recipe.upper()} / {policy_name} / ID {row['image_id']}",
            fill="black",
        )
        image_y = y + label_height
        sheet.paste(thumbnail(clean, thumb_size), (margin, image_y))
        sheet.paste(
            thumbnail(degraded, thumb_size),
            (margin * 2 + thumb_size[0], image_y),
        )
        report_rows.append(
            {
                "image_id": row["image_id"],
                "recipe": recipe,
                "policy": policy_name,
                "rng_seed": seed,
                "haze_level": params.get("haze_level", ""),
                "dark_level": params.get("dark_level", ""),
                "noise_level": params.get("noise_level", ""),
                "transmission_low": params.get("transmission_low", ""),
                "transmission_high": params.get("transmission_high", ""),
                "target_luma_p50": params.get("target_luma_p50", ""),
                "highlight_cap": params.get("highlight_cap", ""),
                "noise_sigma_255": params.get("noise_sigma_255", ""),
            }
        )

    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "pilot_degradation_preview.png"
    sheet.save(path, format="PNG")
    write_csv(
        report_dir / "pilot_degradation_preview_parameters.csv",
        list(report_rows[0]),
        report_rows,
    )
    print("PREVIEW PASSED")
    print(f"Preview: {path}")
    return path


def import_torch_and_unet(repo: Path):
    if not (repo / "models/unet.py").is_file():
        raise FileNotFoundError(f"Senior U-Net source not found: {repo / 'models/unet.py'}")
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        commit = "unknown"
    if commit != EXPECTED_REPO_COMMIT:
        raise RuntimeError(
            f"Senior repository commit mismatch: expected {EXPECTED_REPO_COMMIT}, got {commit}."
        )
    sys.path.insert(0, str(repo))
    import torch
    from models.unet import UNet

    return torch, UNet


def self_test(repo: Path) -> None:
    if np.__version__ != "1.26.4":
        raise RuntimeError(
            f"Expected NumPy 1.26.4 for the pinned environment, got {np.__version__}. "
            "Run install_compatibility.ps1 first."
        )
    height, width = 96, 128
    x = np.linspace(0.05, 0.95, width, dtype=np.float32)
    clean = np.repeat(x[None, :, None], height, axis=0)
    clean = np.repeat(clean, 3, axis=2)
    haze_outputs: dict[str, np.ndarray] = {}
    dark_noise_outputs: dict[str, np.ndarray] = {}
    for policy_name in POLICY_NAMES:
        degradation, components = load_degradation_config(policy_name)
        row = {"record_key": "synthetic/self-test", "recipe": "haze"}
        haze, _, _, haze_params = make_degraded(clean, row, "val", 0, 0, degradation)
        row["recipe"] = "dark_noise"
        dark_noise, _, _, dn_params = make_degraded(clean, row, "val", 1, 0, degradation)
        if haze.shape != clean.shape or dark_noise.shape != clean.shape:
            raise RuntimeError(f"Synthetic shape check failed for {policy_name}.")
        if not np.isfinite(haze).all() or not np.isfinite(dark_noise).all():
            raise RuntimeError(f"Synthetic finite-value check failed for {policy_name}.")
        if np.mean(np.abs(haze - clean)) <= 0.01 or np.mean(np.abs(dark_noise - clean)) <= 0.01:
            raise RuntimeError(f"Synthetic strength check failed for {policy_name}.")
        if haze_params["haze_level"] != "heavy":
            raise RuntimeError(f"Haze mapping failed for {policy_name}.")
        if dn_params["dark_level"] != "heavy" or dn_params["noise_level"] != "heavy":
            raise RuntimeError(f"Dark-noise mapping failed for {policy_name}.")
        print(f"POLICY {policy_name} PASSED: {components}")
        haze_outputs[policy_name] = haze
        dark_noise_outputs[policy_name] = dark_noise

    for policy_name in ("P3", "P4", "P5"):
        if not np.array_equal(haze_outputs["P1"], haze_outputs[policy_name]):
            raise RuntimeError(
                f"Controlled-design failure: P1 and {policy_name} haze must be identical."
            )
    if np.array_equal(haze_outputs["P0"], haze_outputs["P1"]):
        raise RuntimeError("Controlled-design failure: P0 and P1 haze must differ.")
    if np.array_equal(haze_outputs["P1"], haze_outputs["P2"]):
        raise RuntimeError("Controlled-design failure: P1/P2 haze change had no effect.")
    if np.array_equal(dark_noise_outputs["P1"], dark_noise_outputs["P3"]):
        raise RuntimeError("Controlled-design failure: P1/P3 noise change had no effect.")
    if np.array_equal(dark_noise_outputs["P1"], dark_noise_outputs["P4"]):
        raise RuntimeError("Controlled-design failure: P1/P4 darkness change had no effect.")
    print("CONTROLLED-DESIGN PASSED: unchanged components are bit-identical.")

    torch, UNet = import_torch_and_unet(repo)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the F-drive Python environment.")
    device = torch.device("cuda")
    model = UNet(in_channels=3, out_channels=3, base_channels=32).to(device).eval()
    with torch.no_grad():
        tensor = torch.rand(1, 3, 128, 128, device=device)
        output = model(tensor)
    if tuple(output.shape) != (1, 3, 128, 128):
        raise RuntimeError(f"Unexpected U-Net output shape: {tuple(output.shape)}")
    print("SELF-TEST PASSED")
    print(f"PyTorch={torch.__version__}")
    print(f"GPU={torch.cuda.get_device_name(0)}")
    print(f"U-Net output={tuple(output.shape)}")


def paired_crop(
    clean: np.ndarray,
    degraded: np.ndarray,
    crop_size: int,
    rng: np.random.Generator,
    training: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if crop_size <= 0:
        return clean, degraded
    height, width = clean.shape[:2]
    if height < crop_size or width < crop_size:
        raise RuntimeError(f"Crop {crop_size} exceeds image size {width}x{height}.")
    if training:
        top = int(rng.integers(0, height - crop_size + 1))
        left = int(rng.integers(0, width - crop_size + 1))
    else:
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2
    clean = clean[top : top + crop_size, left : left + crop_size]
    degraded = degraded[top : top + crop_size, left : left + crop_size]
    if training and bool(rng.integers(0, 2)):
        clean, degraded = clean[:, ::-1], degraded[:, ::-1]
    if training and bool(rng.integers(0, 2)):
        clean, degraded = clean[::-1, :], degraded[::-1, :]
    return np.ascontiguousarray(clean), np.ascontiguousarray(degraded)


def create_dataset_class(torch):
    class OnlineHeavyDataset(torch.utils.data.Dataset):
        def __init__(
            self,
            rows: list[dict[str, str]],
            degradation: dict[str, Any],
            split: str,
            crop_size: int,
        ):
            self.rows = rows
            self.degradation = degradation
            self.split = split
            self.crop_size = crop_size
            self.epoch = 0

        def set_epoch(self, epoch: int) -> None:
            self.epoch = int(epoch)

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, Any]:
            row = self.rows[index]
            clean = degradation_pipeline.load_rgb_float(Path(row["gt_path"]))
            degraded, recipe, seed, _ = make_degraded(
                clean,
                row,
                self.split,
                index,
                self.epoch,
                self.degradation,
            )
            crop_rng = np.random.default_rng(
                degradation_pipeline.stable_seed(seed, "paired-crop")
            )
            clean, degraded = paired_crop(
                clean,
                degraded,
                self.crop_size,
                crop_rng,
                self.split == "train",
            )
            clean_tensor = torch.from_numpy(clean.transpose(2, 0, 1)).float()
            input_tensor = torch.from_numpy(degraded.transpose(2, 0, 1)).float()
            return {
                "clean": clean_tensor,
                "input": input_tensor,
                "name": row["export_filename"],
                "image_id": row["image_id"],
                "recipe": recipe,
            }

    return OnlineHeavyDataset


def seed_everything(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr_from_mse(mse: float) -> float:
    return float("inf") if mse <= 0 else 10.0 * math.log10(1.0 / mse)


def validate_epoch(torch, model, loader, device) -> dict[str, Any]:
    model.eval()
    total_l1 = 0.0
    input_mse_sum = 0.0
    output_mse_sum = 0.0
    count = 0
    recipe_totals = {
        recipe: {"input_mse_sum": 0.0, "output_mse_sum": 0.0, "count": 0}
        for recipe in ACTIVE_RECIPES
    }
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                output = model(inputs)
                loss = torch.nn.functional.l1_loss(output, clean)
            clipped = output.float().clamp(0.0, 1.0)
            input_mse = ((inputs.float() - clean.float()) ** 2).mean(dim=(1, 2, 3))
            output_mse = ((clipped - clean.float()) ** 2).mean(dim=(1, 2, 3))
            total_l1 += float(loss.item()) * inputs.size(0)
            input_mse_sum += float(input_mse.sum().item())
            output_mse_sum += float(output_mse.sum().item())
            for sample_index, recipe in enumerate(batch["recipe"]):
                if recipe not in recipe_totals:
                    raise RuntimeError(f"Unexpected validation recipe: {recipe}")
                recipe_totals[recipe]["input_mse_sum"] += float(input_mse[sample_index].item())
                recipe_totals[recipe]["output_mse_sum"] += float(output_mse[sample_index].item())
                recipe_totals[recipe]["count"] += 1
            count += inputs.size(0)
    recipe_metrics = {}
    for recipe, totals in recipe_totals.items():
        recipe_count = int(totals["count"])
        if recipe_count <= 0:
            raise RuntimeError(f"Validation recipe has no samples: {recipe}")
        recipe_input_psnr = psnr_from_mse(totals["input_mse_sum"] / recipe_count)
        recipe_output_psnr = psnr_from_mse(totals["output_mse_sum"] / recipe_count)
        recipe_metrics[recipe] = {
            "count": recipe_count,
            "input_psnr_db": recipe_input_psnr,
            "output_psnr_db": recipe_output_psnr,
            "improvement_db": recipe_output_psnr - recipe_input_psnr,
        }
    input_psnr = psnr_from_mse(input_mse_sum / max(count, 1))
    output_psnr = psnr_from_mse(output_mse_sum / max(count, 1))
    return {
        "val_l1": total_l1 / max(count, 1),
        "input_psnr_db": input_psnr,
        "output_psnr_db": output_psnr,
        "improvement_db": output_psnr - input_psnr,
        "per_recipe": recipe_metrics,
    }


def train(root: Path, repo: Path, run_dir: Path, policy_name: str) -> Path:
    ensure_prepared(root, run_dir)
    degradation, policy_components = load_degradation_config(policy_name)
    torch, UNet = import_torch_and_unet(repo)
    seed_everything(torch, SPLIT_SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this pilot run.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    DatasetClass = create_dataset_class(torch)
    train_dataset = DatasetClass(
        read_csv(run_dir / "manifests/pilot_train.csv"),
        degradation,
        "train",
        TRAIN_CROP_SIZE,
    )
    val_dataset = DatasetClass(
        read_csv(run_dir / "manifests/pilot_val.csv"),
        degradation,
        "val",
        VAL_CROP_SIZE,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    model = UNet(in_channels=3, out_channels=3, base_channels=32).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1.0e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    model_dir = run_dir / "models"
    log_dir = run_dir / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_output_psnr = float("-inf")
    best_epoch = 0

    print(f"Model=Senior UNet commit {EXPECTED_REPO_COMMIT}")
    print(f"Trainable parameters={parameter_count:,}")
    print(f"Pilot train={len(train_dataset)}, val={len(val_dataset)}, epochs={PILOT_EPOCHS}")
    print(f"Policy={policy_name}; components={policy_components}")
    print("Training recipes alternate exactly between policy haze and policy dark_noise.")

    training_started = time.perf_counter()
    for epoch in range(PILOT_EPOCHS):
        epoch_started = time.perf_counter()
        train_dataset.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        seen = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            inputs = batch["input"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                output = model(inputs)
                loss = torch.nn.functional.l1_loss(output, clean)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * inputs.size(0)
            seen += inputs.size(0)
            if batch_index % 16 == 0 or batch_index == len(train_loader):
                print(
                    f"epoch {epoch + 1}/{PILOT_EPOCHS} "
                    f"batch {batch_index}/{len(train_loader)} loss={loss.item():.6f}"
                )
        train_loss = running_loss / max(seen, 1)
        validation = validate_epoch(torch, model, val_loader, device)
        val_l1 = float(validation["val_l1"])
        input_psnr = float(validation["input_psnr_db"])
        output_psnr = float(validation["output_psnr_db"])
        haze_metrics = validation["per_recipe"]["haze"]
        dark_noise_metrics = validation["per_recipe"]["dark_noise"]
        row = {
            "epoch": epoch + 1,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "train_l1": train_loss,
            "val_l1": val_l1,
            "val_input_psnr_db": input_psnr,
            "val_output_psnr_db": output_psnr,
            "val_improvement_db": output_psnr - input_psnr,
            "haze_input_psnr_db": haze_metrics["input_psnr_db"],
            "haze_output_psnr_db": haze_metrics["output_psnr_db"],
            "haze_improvement_db": haze_metrics["improvement_db"],
            "dark_noise_input_psnr_db": dark_noise_metrics["input_psnr_db"],
            "dark_noise_output_psnr_db": dark_noise_metrics["output_psnr_db"],
            "dark_noise_improvement_db": dark_noise_metrics["improvement_db"],
        }
        history.append(row)
        write_csv(log_dir / "pilot_training_history.csv", list(history[0]), history)
        state = {
            "params": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "epoch": epoch + 1,
            "tool_version": TOOL_VERSION,
            "policy": policy_name,
            "policy_components": policy_components,
            "senior_repo_commit": EXPECTED_REPO_COMMIT,
            "split_seed": SPLIT_SEED,
            "validation": validation,
        }
        torch.save(state, model_dir / "unet_pilot_last.pth")
        if not math.isfinite(output_psnr):
            raise RuntimeError("Validation PSNR is not finite.")
        if output_psnr > best_output_psnr:
            best_output_psnr = output_psnr
            best_epoch = epoch + 1
            torch.save(state, model_dir / "unet_pilot_best.pth")
        print(
            f"EPOCH {epoch + 1}: train_l1={train_loss:.6f}; val_l1={val_l1:.6f}; "
            f"input_psnr={input_psnr:.4f} dB; output_psnr={output_psnr:.4f} dB; "
            f"delta={output_psnr - input_psnr:+.4f} dB; "
            f"time={row['epoch_seconds']:.1f}s"
        )
        print(
            f"  haze: {haze_metrics['input_psnr_db']:.4f} -> "
            f"{haze_metrics['output_psnr_db']:.4f} dB "
            f"({haze_metrics['improvement_db']:+.4f} dB)"
        )
        print(
            f"  dark_noise: {dark_noise_metrics['input_psnr_db']:.4f} -> "
            f"{dark_noise_metrics['output_psnr_db']:.4f} dB "
            f"({dark_noise_metrics['improvement_db']:+.4f} dB)"
        )

    write_json(
        log_dir / "pilot_training_summary.json",
        {
            "tool_version": TOOL_VERSION,
            "policy": policy_name,
            "policy_components": policy_components,
            "model": "UNet",
            "senior_repo_commit": EXPECTED_REPO_COMMIT,
            "parameter_count": parameter_count,
            "device": torch.cuda.get_device_name(0),
            "epochs": PILOT_EPOCHS,
            "train_count": len(train_dataset),
            "val_count": len(val_dataset),
            "checkpoint_selection": "highest_validation_output_psnr",
            "best_epoch": best_epoch,
            "best_validation_output_psnr_db": best_output_psnr,
            "total_training_seconds": time.perf_counter() - training_started,
            "history": history,
            "purpose": "medium_pilot_before_full_training",
        },
    )
    print("PILOT TRAIN PASSED")
    print(
        f"Best checkpoint selected by validation PSNR: epoch={best_epoch}; "
        f"output_psnr={best_output_psnr:.4f} dB"
    )
    print(f"Best model: {model_dir / 'unet_pilot_best.pth'}")
    return model_dir / "unet_pilot_best.pth"


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8), mode="RGB"
    ).save(path, format="PNG")


def evaluate(root: Path, repo: Path, run_dir: Path, policy_name: str) -> Path:
    ensure_prepared(root, run_dir)
    checkpoint = run_dir / "models/unet_pilot_best.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError("Pilot checkpoint is missing. Run Train first.")
    degradation, policy_components = load_degradation_config(policy_name)
    torch, UNet = import_torch_and_unet(repo)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation.")
    device = torch.device("cuda")
    model = UNet(in_channels=3, out_channels=3, base_channels=32).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if state.get("policy") != policy_name:
        raise RuntimeError(
            f"Checkpoint policy mismatch: expected {policy_name}, got {state.get('policy')}."
        )
    model.load_state_dict(state["params"], strict=True)
    model.eval()

    rows = read_csv(run_dir / "manifests/pilot_val.csv")
    metrics: list[dict[str, Any]] = []
    visual_items: list[tuple[dict[str, str], np.ndarray, np.ndarray, np.ndarray]] = []
    with torch.no_grad():
        for index, row in enumerate(rows):
            clean = degradation_pipeline.load_rgb_float(Path(row["gt_path"]))
            degraded, recipe, seed, _ = make_degraded(
                clean, row, "val", index, 0, degradation
            )
            input_tensor = torch.from_numpy(degraded.transpose(2, 0, 1)).unsqueeze(0).to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                output_tensor = model(input_tensor)
            output = (
                output_tensor[0].float().clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0)
            )
            input_mse = float(np.mean((degraded - clean) ** 2))
            output_mse = float(np.mean((output - clean) ** 2))
            metrics.append(
                {
                    "image_id": row["image_id"],
                    "recipe": recipe,
                    "policy": policy_name,
                    "rng_seed": seed,
                    "input_psnr_db": psnr_from_mse(input_mse),
                    "output_psnr_db": psnr_from_mse(output_mse),
                    "improvement_db": psnr_from_mse(output_mse) - psnr_from_mse(input_mse),
                }
            )
            recipe_visual_count = sum(
                1 for item in visual_items if item[0]["recipe"] == recipe
            )
            if recipe_visual_count < 4:
                visual_items.append((row, clean, degraded, output))

    input_mse_overall = float(
        np.mean([10.0 ** (-float(row["input_psnr_db"]) / 10.0) for row in metrics])
    )
    output_mse_overall = float(
        np.mean([10.0 ** (-float(row["output_psnr_db"]) / 10.0) for row in metrics])
    )
    input_psnr = psnr_from_mse(input_mse_overall)
    output_psnr = psnr_from_mse(output_mse_overall)
    per_recipe = {}
    for recipe in ACTIVE_RECIPES:
        recipe_rows = [row for row in metrics if row["recipe"] == recipe]
        recipe_input_mse = float(
            np.mean(
                [
                    10.0 ** (-float(row["input_psnr_db"]) / 10.0)
                    for row in recipe_rows
                ]
            )
        )
        recipe_output_mse = float(
            np.mean(
                [
                    10.0 ** (-float(row["output_psnr_db"]) / 10.0)
                    for row in recipe_rows
                ]
            )
        )
        recipe_input_psnr = psnr_from_mse(recipe_input_mse)
        recipe_output_psnr = psnr_from_mse(recipe_output_mse)
        per_recipe[recipe] = {
            "count": len(recipe_rows),
            "input_psnr_db": recipe_input_psnr,
            "output_psnr_db": recipe_output_psnr,
            "improvement_db": recipe_output_psnr - recipe_input_psnr,
        }
    report_dir = run_dir / "reports"
    write_csv(report_dir / "pilot_validation_metrics.csv", list(metrics[0]), metrics)
    summary = {
        "sample_count": len(metrics),
        "policy": policy_name,
        "policy_components": policy_components,
        "input_psnr_db": input_psnr,
        "output_psnr_db": output_psnr,
        "improvement_db": output_psnr - input_psnr,
        "checkpoint_epoch": int(state["epoch"]),
        "per_recipe": per_recipe,
        "purpose": "medium_pilot_before_full_training",
    }
    write_json(report_dir / "pilot_validation_summary.json", summary)

    for row, clean, degraded, output in visual_items:
        sample_dir = report_dir / "visual_samples" / row["recipe"]
        stem = f"ID{row['image_id']}"
        save_array(sample_dir / f"{stem}_GT.png", clean)
        save_array(sample_dir / f"{stem}_INPUT.png", degraded)
        save_array(sample_dir / f"{stem}_OUTPUT.png", output)

    thumb_size = (256, 205)
    margin = 14
    label_height = 30
    width = margin * 4 + thumb_size[0] * 3
    height = 42 + len(visual_items) * (label_height + thumb_size[1] + margin)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    headings = (f"{policy_name} INPUT", "U-NET OUTPUT", "CLEAN GT")
    for column, heading in enumerate(headings):
        draw.text((margin + column * (thumb_size[0] + margin), 10), heading, fill="black")
    for row_index, (row, clean, degraded, output) in enumerate(visual_items):
        y = 42 + row_index * (label_height + thumb_size[1] + margin)
        draw.text(
            (margin, y + 6),
            f"{row['recipe'].upper()} / {policy_name} / ID {row['image_id']}",
            fill="black",
        )
        image_y = y + label_height
        for column, array in enumerate((degraded, output, clean)):
            sheet.paste(
                thumbnail(array, thumb_size),
                (margin + column * (thumb_size[0] + margin), image_y),
            )
    grid_path = report_dir / "pilot_unet_input_output_gt.png"
    sheet.save(grid_path, format="PNG")
    print("PILOT EVALUATION PASSED")
    print(f"INPUT PSNR={input_psnr:.4f} dB")
    print(f"OUTPUT PSNR={output_psnr:.4f} dB")
    print(f"IMPROVEMENT={output_psnr - input_psnr:+.4f} dB")
    for recipe in ACTIVE_RECIPES:
        recipe_metrics = per_recipe[recipe]
        print(
            f"{recipe}: INPUT={recipe_metrics['input_psnr_db']:.4f} dB; "
            f"OUTPUT={recipe_metrics['output_psnr_db']:.4f} dB; "
            f"IMPROVEMENT={recipe_metrics['improvement_db']:+.4f} dB"
        )
    print("This is a 512/80/10-epoch pilot, not the final experiment or test-set result.")
    print(f"Result grid: {grid_path}")
    return grid_path


def create_policy_visual_comparison(run_parent: Path, recipe: str) -> Path:
    reference_dir = run_parent / "P0/reports/visual_samples" / recipe
    gt_paths = sorted(reference_dir.glob("ID*_GT.png"))
    if len(gt_paths) != 4:
        raise FileNotFoundError(
            f"Expected four {recipe} visual samples from P0. Re-run Evaluate for P0."
        )

    sample_ids = [path.name.removesuffix("_GT.png") for path in gt_paths]
    for policy_name in POLICY_NAMES:
        sample_dir = run_parent / policy_name / "reports/visual_samples" / recipe
        for sample_id in sample_ids:
            for role in ("GT", "INPUT", "OUTPUT"):
                path = sample_dir / f"{sample_id}_{role}.png"
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing visual sample: {path}. Re-run Evaluate for {policy_name}."
                    )
            if sha256_file(sample_dir / f"{sample_id}_GT.png") != sha256_file(
                reference_dir / f"{sample_id}_GT.png"
            ):
                raise RuntimeError(
                    f"GT mismatch across policies for {recipe}/{sample_id}."
                )

    thumb_size = (160, 128)
    margin = 10
    label_height = 28
    headings = ["CLEAN GT"]
    for policy_name in POLICY_NAMES:
        headings.extend((f"{policy_name} INPUT", f"{policy_name} OUTPUT"))
    column_count = len(headings)
    width = margin * (column_count + 1) + thumb_size[0] * column_count
    row_height = label_height + thumb_size[1] + margin
    height = 42 + len(sample_ids) * row_height
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for column, heading in enumerate(headings):
        x = margin + column * (thumb_size[0] + margin)
        draw.text((x, 10), heading, fill="black")

    for row_index, sample_id in enumerate(sample_ids):
        y = 42 + row_index * row_height
        draw.text((margin, y + 6), sample_id, fill="black")
        image_y = y + label_height
        image_paths = [reference_dir / f"{sample_id}_GT.png"]
        for policy_name in POLICY_NAMES:
            sample_dir = run_parent / policy_name / "reports/visual_samples" / recipe
            image_paths.extend(
                (
                    sample_dir / f"{sample_id}_INPUT.png",
                    sample_dir / f"{sample_id}_OUTPUT.png",
                )
            )
        for column, path in enumerate(image_paths):
            with Image.open(path) as source:
                array = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
            x = margin + column * (thumb_size[0] + margin)
            sheet.paste(thumbnail(array, thumb_size), (x, image_y))

    report_dir = run_parent / "comparison"
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / f"P0_P5_{recipe}_visual_comparison.png"
    sheet.save(output, format="PNG")
    return output


def compare_policies(run_parent: Path) -> Path:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for policy_name in POLICY_NAMES:
        path = run_parent / policy_name / "reports/pilot_validation_summary.json"
        if not path.is_file():
            missing.append(policy_name)
            continue
        with path.open("r", encoding="utf-8-sig") as handle:
            summary = json.load(handle)
        if summary.get("policy") != policy_name:
            raise RuntimeError(f"Summary policy mismatch in {path}.")
        haze = summary["per_recipe"]["haze"]
        dark_noise = summary["per_recipe"]["dark_noise"]
        components = summary["policy_components"]
        rows.append(
            {
                "policy": policy_name,
                "haze_level": components["haze"],
                "dark_level": components["dark"],
                "noise_level": components["noise"],
                "overall_input_psnr_db": summary["input_psnr_db"],
                "overall_output_psnr_db": summary["output_psnr_db"],
                "overall_improvement_db": summary["improvement_db"],
                "haze_input_psnr_db": haze["input_psnr_db"],
                "haze_output_psnr_db": haze["output_psnr_db"],
                "haze_improvement_db": haze["improvement_db"],
                "dark_noise_input_psnr_db": dark_noise["input_psnr_db"],
                "dark_noise_output_psnr_db": dark_noise["output_psnr_db"],
                "dark_noise_improvement_db": dark_noise["improvement_db"],
                "checkpoint_epoch": summary["checkpoint_epoch"],
            }
        )
    if missing:
        raise FileNotFoundError(
            "Evaluate these policies before Compare: " + ", ".join(missing)
        )
    report_dir = run_parent / "comparison"
    path = report_dir / "P0_P5_validation_comparison.csv"
    write_csv(path, list(rows[0]), rows)
    write_json(
        report_dir / "P0_P5_validation_comparison.json",
        {
            "tool_version": TOOL_VERSION,
            "policies": rows,
            "selection_warning": (
                "Do not choose a policy from improvement alone. Consider absolute output "
                "PSNR, both recipe results, and vehicle-detail inspection."
            ),
        },
    )
    haze_visual = create_policy_visual_comparison(run_parent, "haze")
    dark_noise_visual = create_policy_visual_comparison(run_parent, "dark_noise")
    print("COMPARISON PASSED")
    for row in rows:
        print(
            f"{row['policy']} {row['haze_level']}+{row['dark_level']}{row['noise_level']}: "
            f"overall {float(row['overall_input_psnr_db']):.4f} -> "
            f"{float(row['overall_output_psnr_db']):.4f} dB; "
            f"haze out={float(row['haze_output_psnr_db']):.4f}; "
            f"dark_noise out={float(row['dark_noise_output_psnr_db']):.4f}"
        )
    print(f"Comparison CSV: {path}")
    print(f"Haze visual comparison: {haze_visual}")
    print(f"Dark-noise visual comparison: {dark_noise_visual}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=("prepare", "preview", "self-test", "train", "evaluate", "compare", "all"),
        required=True,
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--policy", choices=POLICY_NAMES, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    repo = args.repo.resolve()
    run_dir = args.run_dir.resolve()
    policy_name = args.policy
    run_dir.mkdir(parents=True, exist_ok=True)
    action = args.action
    if action != "compare":
        write_policy_snapshot(run_dir, policy_name)
    if action == "prepare":
        prepare(root, run_dir)
    elif action == "preview":
        preview(root, run_dir, policy_name)
    elif action == "self-test":
        self_test(repo)
    elif action == "train":
        train(root, repo, run_dir, policy_name)
    elif action == "evaluate":
        evaluate(root, repo, run_dir, policy_name)
    elif action == "compare":
        compare_policies(run_dir.parent)
    elif action == "all":
        prepare(root, run_dir)
        self_test(repo)
        preview(root, run_dir, policy_name)
        train(root, repo, run_dir, policy_name)
        evaluate(root, repo, run_dir, policy_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
