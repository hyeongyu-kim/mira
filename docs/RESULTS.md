# Results

## CNN reference

Source: camera-ready Tables 2 and 26. Values are mean ± sample SD (`ddof=1`).
Accuracy is in percent; accuracy SD is in percentage points.

| Dataset / source | Method | Accuracy (%) | MMC | ECE | NLL |
|---|---|---:|---:|---:|---:|
| CIFAR-100-C / ResNeXt AugMix | TENT | 67.84 ± 0.05 | 0.7461 ± 0.0007 | 0.0693 ± 0.0004 | 1.2547 ± 0.0021 |
| CIFAR-100-C / ResNeXt AugMix | TENT+MIRA | 68.69 ± 0.15 | 0.7305 ± 0.0012 | 0.0452 ± 0.0007 | 1.1940 ± 0.0111 |
| ImageNet-C / ResNet-50 V1 | TENT | 36.62 ± 0.12 | 0.4089 ± 0.0014 | 0.0478 ± 0.0018 | 3.2549 ± 0.0075 |
| ImageNet-C / ResNet-50 V1 | TENT+MIRA | 39.45 ± 0.14 | 0.4053 ± 0.0014 | 0.0361 ± 0.0022 | 3.0205 ± 0.0020 |

## ViT reference

Profile: `vit_b16_torchvision_batch64_v2`. ImageNet-C, torchvision ViT-B/16,
learning rate 0.00025, batch size 64.

| Method | Accuracy (%) | MMC | ECE | NLL |
|---|---:|---:|---:|---:|
| TENT | 41.64 | 0.5170 | 0.1167 | 3.0582 |
| TENT+MIRA | 44.76 | 0.4908 | 0.0549 | 2.8406 |

## Experiment settings

| Experiment | Dataset / backbone | Settings |
|---|---|---|
| Main CNN | CIFAR-100-C / ResNeXt AugMix | SGD, LR 0.001, batch 64, K=300 |
| Main CNN | ImageNet-C / ResNet-50 | SGD, LR 0.00025, batch 64, K=300 |
| Main ViT | ImageNet-C / torchvision ViT-B/16 | SGD, LR 0.00025, batch 64, K=300 |
| Objective / gate, Tables 1 and 6 | ImageNet-C / ResNet-34 | SGD, LR 0.00025 |
| Gradient norm, Table 7 | CIFAR-10-C / WRN, CIFAR-100-C / ResNeXt, ImageNet-C / ResNet-50 | Global adaptation-gradient L2 norm = 1 |
| Objective / gate, Table 8 | CIFAR-10-C / WRN AugMix | Adam, LR 0.001, K=100; gaussian noise, snow, fog, pixelation |
| Perturbation granularity, Table 9 | CIFAR-100-C / WRN AugMix | Feature / scalar / class; He-in; K=100 |
| Perturbation scale, Table 10 | CIFAR-100-C / WRN AugMix | He-in / fan-in / absolute / weight magnitude; feature-wise; K=100 |
| α–ρ, Table 13 | CIFAR-100-C / ResNeXt, ImageNet-C / ResNet-50 | α ∈ {0.01, 0.03, 0.10}; ρ ∈ {0.85, 0.90, 0.95}; K=300 |
| γ, Table 14 | CIFAR-100-C / ResNeXt, ImageNet-C / ResNet-50 | γ ∈ {0, 0.25, 0.5, 1, 2}; HE + γ·EH; K=300 |
| σ grid, Tables 15 and 16 | CIFAR-100-C / ResNeXt, ImageNet-C / ResNet-50 | Original / coarse / dense / extended; K=300 |

Main evaluation: severity 5, 15 corruptions, 10,000 images per corruption,
one update per batch, model and optimizer reset per corruption.

[Configs](../code/configs/) · [Setup](SETUP.md)

## Result files

Paths are relative to [code/results](../code/results/).

| File | Contents |
|---|---|
| `seed_results.csv`, `mean_sd_summary.csv` | CNN reference metrics |
| `vit_seed_results.csv`, `vit_mean_sd_summary.csv`, `vit_results.tex` | Selected ViT metrics and LaTeX table |
| `archive/vit_timm_batch64_v1/`, `archive/vit_batch128/` | Previous ViT profile metrics |
