# MIRA implementation

See the [repository README](../README.md) for installation and reference results.
Run the commands below from this directory after the [asset setup](../docs/SETUP.md):

```bash
bash scripts/run_main.sh
bash scripts/run_vit.sh
bash scripts/run_ablations.sh
python scripts/validate_release.py
python -m unittest discover -s tests -v
```

[Results](../docs/RESULTS.md) · [Reproducibility](../docs/REPRODUCIBILITY.md) ·
[Changing the ViT version](../docs/VIT_PROFILES.md) ·
[Paper table commands](docs/table_commands.json).
