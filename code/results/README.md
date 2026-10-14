# Result files

| Files | Contents |
|---|---|
| `seed_results.csv`, `mean_sd_summary.csv`, `paired_seed_differences.csv` | CNN reference |
| `release_validation_*.csv` | CNN validation |
| `vit_*.csv`, `vit_results.tex` | Selected torchvision ViT |
| `archive/vit_timm_batch64_v1/` | timm batch-64 profile |
| `archive/vit_batch128/` | timm batch-128 profile |
| `diagnostics/index.json` | Archive hashes and file counts |
| `batch_log_export_manifest.json` | Batch-log file index |

Extract archives from `code/`:

```bash
mkdir -p /path/to/diagnostics
tar -xzf results/diagnostics/cnn_reference.tar.gz -C /path/to/diagnostics
tar -xzf results/diagnostics/cnn_release_checks.tar.gz -C /path/to/diagnostics
tar -xzf results/diagnostics/vit_torchvision_batch64_v2.tar.gz -C /path/to/diagnostics
tar -xzf results/diagnostics/vit_timm_batch64_v1.tar.gz -C /path/to/diagnostics
tar -xzf results/diagnostics/vit_batch128.tar.gz -C /path/to/diagnostics
```

[Metrics and settings](../../docs/RESULTS.md) · [Validation](../../docs/REPRODUCIBILITY.md)
