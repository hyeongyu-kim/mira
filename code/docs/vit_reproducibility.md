# ViT release reference

The selected reference is torchvision ViT-B/16, LR .00025 / batch64 / seed1:
TENT 41.64 / .5170 / .1167 / 3.0582 and TENT+MIRA 44.76 / .4908 / .0549 / 2.8406
(Accuracy %, MMC, ECE, NLL).

See [results and protocol](../../docs/RESULTS.md),
[versioned profiles](../../docs/VIT_PROFILES.md) and
[validation coverage](../../docs/REPRODUCIBILITY.md).
The default executes the measured seed1 pair; seeds2/3 are prepared, unmeasured configs.
Previous timm batch64 and batch128 evidence is kept in separate `results/archive/` directories.
The original Table3 42.03/45.17 rows have recovered LR/batch settings, but not complete
checkpoint/seed/stream provenance. The measured accuracies differ by −.39/−.41 pp.
Fallback is 99.62%; calibration improvement is not evidence of unconditional agreement.
