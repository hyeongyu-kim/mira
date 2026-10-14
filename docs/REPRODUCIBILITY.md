# Validation

## Validation counts

| Item | Count / value |
|---|---:|
| Source and config files | 334 |
| Portable run configs | 235 |
| Archived CNN runs | 217 |
| CNN statistical groups | 73 |
| Paper numerical cells | 884 |
| CPU tests | 22 |
| Full CNN validation runs | 13 |
| Full selected-ViT runs | 2 |
| Matched CNN corruption streams | 90 |
| Matched selected-ViT corruption streams | 15 |
| Maximum gradient-norm error | 0.000248432 |

Paper: `NeurIPS_2026_MIRA_camera_ready_v1.pdf`.
Tables: 1, 2 (TENT/MIRA), 6–10, 13–16, 26 (TENT/MIRA), 27.

[Cell audit](paper_numeric_audit.csv) · [PDF hash and audit metadata](paper_numeric_audit.json) ·
[Table configs](../code/docs/table_commands.json)

## CNN accuracy

Values are mean ± sample SD; accuracy is in percent and SD in percentage points.

| Dataset / backbone | Method | Fresh accuracy (%) | Archived accuracy (%) |
|---|---|---:|---:|
| CIFAR-100-C / ResNeXt | TENT | 67.85 ± 0.05 | 67.84 ± 0.05 |
| CIFAR-100-C / ResNeXt | TENT+MIRA | 68.70 ± 0.16 | 68.69 ± 0.15 |
| ImageNet-C / ResNet-50 | TENT | 36.62 ± 0.10 | 36.62 ± 0.12 |
| ImageNet-C / ResNet-50 | TENT+MIRA | 39.45 ± 0.14 | 39.45 ± 0.14 |

| Dataset / backbone | Maximum absolute run difference (pp) |
|---|---:|
| CIFAR-100-C / ResNeXt | 0.012000 |
| ImageNet-C / ResNet-50 | 0.018667 |
| ImageNet-C / ResNet-34 | 0.026001 |

[Validation CSVs](../code/results/) · [Selected ViT metrics](RESULTS.md#vit-reference)

## Checks

Run from `code/`:

```bash
python scripts/validate_release.py
python -m unittest discover -s tests -v
python scripts/check_environment.py
python scripts/check_assets.py --scope all --check-data
bash scripts/run_vit.sh --dry-run
```

| Record | File |
|---|---|
| Source and checkpoint hashes | [reference_manifest.json](../code/reference_manifest.json) |
| CNN validation | [release_validation_vs_archived.csv](../code/results/release_validation_vs_archived.csv) |
| Selected ViT command checks | [torchvision_profile_validation.json](../code/docs/validation_checks/torchvision_profile_validation.json) |
| Selected ViT full runs | [torchvision_vit_full_validation.json](../code/docs/validation_checks/torchvision_vit_full_validation.json) |
| CPU test output | [torchvision_cpu_tests.txt](../code/docs/validation_checks/torchvision_cpu_tests.txt) |
| Archive hashes | [diagnostics/index.json](../code/results/diagnostics/index.json) |

GitHub Actions: CPU tests, source/config/statistics checks and ViT command selection.
