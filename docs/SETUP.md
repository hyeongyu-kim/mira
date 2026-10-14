# Setup

## Environment

| Package | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.2.1 |
| torchvision | 0.17.1 |
| CUDA | 12.1 |
| timm | 1.0.15 |
| NumPy | 1.26.4 |
| Pillow | 10.2.0 |

[Installation commands](../README.md#installation) ·
[Conda lock](../code/environment-conda-linux-64.lock.txt) ·
[Pip requirements](../code/requirements-reference.txt)

## Data

Download [CIFAR-10-C](https://zenodo.org/records/2535967),
[CIFAR-100-C](https://zenodo.org/records/3555552) and
[ImageNet-C](https://zenodo.org/records/2235448).

```text
corruption-data/
  CIFAR-10-C/labels.npy
  CIFAR-10-C/gaussian_noise.npy
  ...
  CIFAR-100-C/labels.npy
  CIFAR-100-C/gaussian_noise.npy
  ...
  ImageNet-C/gaussian_noise/5/n01440764/ILSVRC2012_val_00000293.JPEG
  ...
```

ImageNet-C input: 224×224 RGB images with official validation filenames.

## Checkpoints

Paths below are relative to `MIRA_CHECKPOINT_ROOT`.
Hashes and file sizes: [reference_manifest.json](../code/reference_manifest.json).

| Path | Download |
|---|---|
| `cifar100/corruptions/Hendrycks2020AugMix_ResNeXt.pt` | RobustBench GDrive ID `1ocnHbvDdOBLvgNr6K7vEYL08hUdkD1Rv` |
| `cifar100/corruptions/Hendrycks2020AugMix_WRN.pt` | RobustBench GDrive ID `1XpFFdCdU9LcDtcyNfo6_BV1RZHKKkBVE` |
| `cifar10/corruptions/Hendrycks2020AugMix_WRN.pt` | RobustBench GDrive ID `1wy7gSRsUZzCzj8QhmTbcnwmES_2kkNph` |
| `resnet50-0676ba61.pth` | [torchvision](https://download.pytorch.org/models/resnet50-0676ba61.pth) |
| `resnet34-b627a593.pth` | [torchvision](https://download.pytorch.org/models/resnet34-b627a593.pth) |
| `vit_b_16-c867db91.pth` | [torchvision](https://download.pytorch.org/models/vit_b_16-c867db91.pth) |
| `jx_vit_base_p16_224-80ecf9dd.pth` | [timm](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth) |

## Run

From `code/`:

```bash
export MIRA_DATA_ROOT=/path/to/corruption-data
export MIRA_CHECKPOINT_ROOT=/path/to/source-checkpoints
export MIRA_RESULTS_ROOT=/path/to/results
export CUDA_VISIBLE_DEVICES=0

python scripts/check_assets.py --scope main --check-data
python scripts/check_assets.py --scope vit --check-data
bash scripts/run_main.sh
bash scripts/run_vit.sh
bash scripts/run_ablations.sh
python scripts/summarize.py --runs "$MIRA_RESULTS_ROOT" --output results_local
```

Individual run and two-batch smoke check:

```bash
python scripts/run_reference.py --config configs/c100_full_s1.json
MIRA_RESULTS_ROOT=/path/to/smoke-results \
  python scripts/run_reference.py --config configs/vit_b16_full_s1.json --max-batches 2
```

Rerun the same command to reuse completed runs or resume at saved corruption boundaries.

## Evaluation settings

| Setting | Main recipes |
|---|---|
| Severity | 5 |
| Corruptions | 15 |
| Images per corruption | 10,000 |
| Batch size | 64 |
| Updates per batch | 1 |
| Reset | Model and optimizer per corruption |
| Optimizer | SGD; momentum 0.9; Nesterov; weight decay 0 |
| Predictions | Before update |
| Accuracy | Percent; macro mean over corruptions |
| MMC / ECE / NLL | Macro mean over corruptions |
| ECE bins | 10 equal-width |
| SD | Sample SD, ddof=1; accuracy SD in percentage points |

```text
gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur,
motion_blur, zoom_blur, snow, frost, fog, brightness, contrast,
elastic_transform, pixelate, jpeg_compression
```

[Model and ablation settings](RESULTS.md#experiment-settings) · [MIRA defaults](CODE_SEMANTICS.md)

## Outputs

Each run writes configs, checkpoint/source hashes, metrics, batch logs, runtime and memory.
Complete results: `result.json`. Batch logs: `*.batches.jsonl`.
