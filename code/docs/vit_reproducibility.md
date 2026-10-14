# ViT reference

| Setting | Value |
|---|---|
| Profile | `vit_b16_torchvision_batch64_v2` |
| Dataset | ImageNet-C |
| Backbone | torchvision ViT-B/16 |
| Learning rate | 0.00025 |
| Batch size | 64 |

| Method | Accuracy (%) | MMC | ECE | NLL |
|---|---:|---:|---:|---:|
| TENT | 41.64 | 0.5170 | 0.1167 | 3.0582 |
| TENT+MIRA | 44.76 | 0.4908 | 0.0549 | 2.8406 |

[Settings and commands](../../docs/VIT_PROFILES.md) ·
[Validation records](../../docs/REPRODUCIBILITY.md) ·
[Previous profile metrics](../results/archive/)
