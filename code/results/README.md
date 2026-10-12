# Reference results

The small seed, mean/sample-SD and paired-difference CSVs are directly browsable.
CNN reference, fresh validation, current ViT and previous batch128 ViT stay separate.
Detailed per-run configurations, provenance, per-corruption records and batch diagnostics
are grouped into four verified archives under `diagnostics/`.

```bash
mkdir -p /path/to/diagnostics
tar -xzf results/diagnostics/cnn_reference.tar.gz -C /path/to/diagnostics
tar -xzf results/diagnostics/cnn_release_checks.tar.gz -C /path/to/diagnostics
tar -xzf results/diagnostics/vit_batch64.tar.gz -C /path/to/diagnostics
tar -xzf results/diagnostics/vit_batch128.tar.gz -C /path/to/diagnostics
```

`diagnostics/index.json` records archive hashes, original paths and counts. All archived
files were checked against their original SHA256 before the loose copies were removed.
`batch_log_export_manifest.json` describes the original paths inside these archives.
See the root `docs/RESULTS.md` and `docs/REPRODUCIBILITY.md` for protocols and scope.
