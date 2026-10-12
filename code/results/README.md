# Reference results

The small seed, mean/sample-SD and paired-difference CSVs are directly browsable.
CNN reference, fresh validation, current torchvision ViT and previous timm ViT stay separate.
Detailed per-run configurations, provenance, per-corruption records and batch diagnostics
are grouped into five verified archives under `diagnostics/`.

```bash
mkdir -p /path/to/diagnostics
tar -xzf results/diagnostics/cnn_reference.tar.gz -C /path/to/diagnostics
tar -xzf results/diagnostics/cnn_release_checks.tar.gz -C /path/to/diagnostics
tar -xzf results/diagnostics/vit_torchvision_batch64_v2.tar.gz -C /path/to/diagnostics
tar -xzf results/diagnostics/vit_timm_batch64_v1.tar.gz -C /path/to/previous-timm-diagnostics
tar -xzf results/diagnostics/vit_batch128.tar.gz -C /path/to/diagnostics
```

`diagnostics/index.json` records archive hashes, original paths and counts. Earlier archives'
files were checked against their original SHA256 before the loose copies were removed.
The new torchvision archive records original and exported file hashes separately because
host paths are made portable and redundant per-feature vectors are omitted from exports.
`batch_log_export_manifest.json` describes the original paths inside these archives.
See the root `docs/RESULTS.md` and `docs/REPRODUCIBILITY.md` for protocols and scope.
