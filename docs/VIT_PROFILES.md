# ViT profiles

Registry: [vit_profiles.json](../code/configs/vit_profiles.json).
Default: `vit_b16_torchvision_batch64_v2`.

| Profile | Backbone | LR | Batch |
|---|---|---:|---:|
| `vit_b16_torchvision_batch64_v2` | torchvision ViT-B/16 V1 | 0.00025 | 64 |
| `vit_b16_batch64_v1` | timm ViT-B/16 orig_in21k_ft_in1k | 0.001 | 64 |
| `vit_b16_batch128_previous` | timm ViT-B/16 orig_in21k_ft_in1k | 0.001 | 128 |

| Library | Checkpoint | Input mean | Input std |
|---|---|---|---|
| torchvision | `vit_b_16-c867db91.pth` | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) |
| timm | `jx_vit_base_p16_224-80ecf9dd.pth` | (0.5, 0.5, 0.5) | (0.5, 0.5, 0.5) |

ImageNet-C, 224×224 RGB, severity 5, 15 corruptions, 10,000 images per corruption,
one update per batch, reset per corruption. Optimizer: SGD, momentum 0.9,
Nesterov, weight decay 0. Adaptation: LayerNorm weight and bias.

## Run

From `code/`:

```bash
bash scripts/run_vit.sh
bash scripts/run_vit.sh --methods full
bash scripts/run_vit.sh --list-profiles
bash scripts/run_vit.sh --dry-run
bash scripts/run_vit.sh --profile vit_b16_batch64_v1
bash scripts/run_vit.sh --profile vit_b16_batch128_previous
```

## Change the default

Add paired configs with new run IDs, register the profile in `configs/vit_profiles.json`,
and set `default_profile` to the new profile. Update its `reference_results` CSV path.

```bash
python scripts/update_release_manifest.py
python scripts/validate_release.py
bash scripts/run_vit.sh --dry-run
```

[Selected metrics](RESULTS.md#vit-reference) · [Previous metrics](../code/results/archive/)
