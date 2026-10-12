# ViT release reference

The selected reference is batch64 seed1: TENT 45.96 / .7330 / .2736 / 4.0038 and
TENT+MIRA 50.02 / .6393 / .1410 / 2.9037 (Accuracy %, MMC, ECE, NLL).

See [results and protocol](../../docs/RESULTS.md),
[versioned profiles](../../docs/VIT_PROFILES.md) and
[validation coverage](../../docs/REPRODUCIBILITY.md).
The default executes the measured seed1 pair; seeds2/3 are prepared, unmeasured configs.
Previous batch128 three-seed evidence is kept in `results/archive/vit_batch128/`.
The original Table3 42.03/45.17 config was not recovered; neither profile is labeled its exact reproduction.
