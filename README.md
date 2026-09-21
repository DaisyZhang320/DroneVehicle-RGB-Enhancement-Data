# DroneVehicle 可见光增强数据与退化测试集

本仓库整理了 DroneVehicle 可见光图像增强实验使用的干净图像、退化脚本，以及一套固定的六组退化测试输入。测试 INPUT 与干净 GT 一一对应，可供不同增强模型在相同测试集上评估。

## 仓库内容

| 路径 | 内容 |
| --- | --- |
| `clean_gt/train/` | 人工筛选的 3967 张白天干净训练 GT，按图像 ID 前缀分目录保存。 |
| `clean_gt/test/` | 固定的 200 张干净测试 GT，以 `{image_id}.png` 命名。 |
| `manifests/` | 干净测试 GT 的索引及 SHA-256 校验信息。 |
| `test200_six_groups_v1/INPUT/` | 对应固定测试 GT 的 200 张退化 INPUT，分为六组。 |
| `test200_six_groups_v1/test_pairs.csv` | 每张 INPUT 与 GT 的对应关系、组别、随机种子、退化参数及文件校验值。 |
| `scripts/` | 退化预览、U-Net 小规模试验和早期测试配对生成脚本。 |

干净 GT 按有效图像区域裁剪，以无损 PNG 格式保存；原始 DroneVehicle 图像未被修改。训练 GT 和测试 GT 分开存放。

## 六组固定测试输入

使用同一批 **200 张测试 GT**，每张 GT 只分配到以下六组中的一组，并生成一张退化 INPUT。H 表示雾霾，D 表示暗光，N 表示噪声。

| 分组 | 退化类型 | 张数 | 使用的参数范围 |
| --- | --- | ---: | --- |
| H1 | 雾霾 | 16 | P0 试验中的 H1 |
| H2 | 雾霾 | 18 | P1 试验中的 H2 |
| H3 | 雾霾 | 16 | P2 试验中的 H3 |
| D1N1 | 暗光＋噪声 | 50 | P0 试验中的 D1、N1 |
| D1N2 | 暗光＋噪声 | 50 | P3 试验中的 D1、N2 |
| D2N1 | 暗光＋噪声 | 50 | P4 试验中的 D2、N1 |
| **合计** | 雾霾 50 张；暗光＋噪声 150 张 | **200** | |

P0–P4 在这里用于说明退化**参数范围**。此前使用 U-Net 进行的小规模试验结果，不能直接视为其他模型的恢复边界。每张图像实际采样到的参数记录在 `test_pairs.csv` 的 `parameters_json` 字段中。

## 如何读取 INPUT—GT 配对

在仓库根目录运行以下 Python 示例：

```python
import csv
from pathlib import Path

repo = Path.cwd()
manifest = repo / "test200_six_groups_v1/test_pairs.csv"

with manifest.open(encoding="utf-8-sig", newline="") as file:
    for row in csv.DictReader(file):
        input_image = repo / "test200_six_groups_v1" / row["input_relpath"]
        clean_gt = repo / row["gt_path"]

        assert input_image.is_file() and clean_gt.is_file()
        print(row["image_id"], row["group"], input_image, clean_gt)
```

清单中的 `gt_path` 指向 `clean_gt/test/{image_id}.png`；`input_relpath` 相对于 `test200_six_groups_v1/`。`gt_sha256` 和 `input_sha256` 可用于核对文件，`rng_seed` 和 `parameters_json` 记录退化生成信息。

评估模型时，建议分别报告六组结果。六组的样本数不同，仅报告一个总体平均值不足以说明模型在各类退化下的表现。

## 已上传脚本

| 路径 | 用途 |
| --- | --- |
| `scripts/degradation_preview_v1/` | 预览不同雾霾、暗光、噪声强度，辅助确定参数范围。 |
| `scripts/limit_pilot_v1/` | P0–P5 的 U-Net 小规模试验，包括退化预览、训练和验证程序。 |
| `scripts/test200_builder_v2/` | 较早的两类退化测试配对生成程序。 |

**复现说明：**仓库已包含本次六组测试的 INPUT、GT 和逐图退化参数。目前尚未上传这次六组测试的专用生成脚本及其依赖的 `candidate_policy.json`、`config.json`。因此可以直接使用和核对这套固定测试集，但不能仅凭仓库现有脚本一键重新生成完全相同的六组 INPUT。`scripts/test200_builder_v2/` 不是本次六组输入的生成器。

## 原始数据来源

原始数据集：[DroneVehicle](https://github.com/VisDrone/DroneVehicle)。使用本仓库数据时，请注明原始数据来源，并核对原始数据集的使用说明。
