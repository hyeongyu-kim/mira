# MIRA

**MIRA: Mutual Information guided calibration for Reliable Test-Time Adaptation**

Hyeongyu Kim, Mingyeong Jang, Youngjun Song, GeonHui Han, Dosik Hwang · NeurIPS 2026

[Paper / OpenReview](https://openreview.net/forum?id=ujUYtJK5LG) ·
[Project page](https://hyeongyu-kim.github.io/mira/) ·
[Results and protocols](docs/RESULTS.md) · [Reproducibility](docs/REPRODUCIBILITY.md) · [Code/paper semantics](docs/CODE_SEMANTICS.md)

This repository provides **TENT and TENT+MIRA for CNNs and ViT-B/16**, together with
objective, reliability-gating and perturbation controls. MIRA probes cached features with
classifier perturbations, selects a radius using mutual information and prediction agreement,
and adapts normalization parameters through a weighted stochastic entropy objective.

<!-- Overview figure reserved. Add the final figure here when ready. -->

## Installation

The reference environment is Python 3.12, PyTorch 2.2.1, torchvision 0.17.1, CUDA 12.1,
timm 1.0.15 and NumPy 1.26.4. A Linux conda lock and pip version pins are included.

```bash
git clone https://github.com/hyeongyu-kim/mira.git
cd mira
conda create -n mira --file code/environment-conda-linux-64.lock.txt
conda activate mira
python -m pip install -r code/requirements-reference.txt
python -m pip install -e code --no-deps
cd code
```

Prepare the corruption datasets and source weights as described in [Setup](docs/SETUP.md).
Weights are identified by full SHA256 hashes in
[the reference manifest](code/reference_manifest.json); evaluation does not download or
replace them automatically.

## Run

| Reference protocol | Batch | Learning rate | Measured seeds |
|---|---:|---:|---|
| CIFAR-100-C / ResNeXt AugMix | 64 | 0.001 | 1, 2, 3 |
| ImageNet-C / ResNet-50 V1 | 64 | 0.00025 | 1, 2, 3 |
| ImageNet-C / torchvision ViT-B/16 V1 | 64 | 0.00025 | 1 |

All three use severity 5, 15 corruptions, 10,000 examples per corruption, one adaptation
step per batch, and model/optimizer reset at every corruption boundary. Metrics use
predictions made before the update. ECE uses 10 equal-width bins.

```bash
# From code/; choose the local data, weight and output paths.
export MIRA_DATA_ROOT=/path/to/corruption-data
export MIRA_CHECKPOINT_ROOT=/path/to/source-checkpoints
export MIRA_RESULTS_ROOT=/path/to/results
export CUDA_VISIBLE_DEVICES=0

python scripts/check_assets.py --scope all --check-data
bash scripts/run_main.sh        # CNN: paired TENT/MIRA, seeds 1/2/3
bash scripts/run_vit.sh         # current ViT: paired TENT/MIRA, seed 1
python scripts/summarize.py --runs "$MIRA_RESULTS_ROOT" --output results_local
```

`bash run_all.sh main` runs both families. Each run starts from its source checkpoint;
matching completed evaluations are reused, and interrupted evaluations resume at saved
corruption boundaries. Configs, source/checkpoint hashes, parameter lists, per-corruption
metrics, and batch sigma/MI/agreement/fallback diagnostics are saved alongside the results.

The ViT default is versioned in [vit_profiles.json](code/configs/vit_profiles.json).
Additional repeats and previous timm validations are explicit choices:

```bash
bash scripts/run_vit.sh --list-profiles
bash scripts/run_vit.sh --dry-run
bash scripts/run_vit.sh --seeds 1 2 3
bash scripts/run_vit.sh --profile vit_b16_batch64_v1
bash scripts/run_vit.sh --profile vit_b16_batch128_previous
```

See [ViT profiles](docs/VIT_PROFILES.md) for changing the reference without overwriting old
results. Seeds 2/3 of the current batch-64 profile are prepared but have not been measured.

```bash
bash scripts/run_ablations.sh   # all supported CNN objective/construction/sensitivity profiles
python -m unittest discover -s tests -v
python scripts/validate_release.py
```

[Table-to-command mapping](code/docs/table_commands.json) identifies the supported paper
comparisons. Global gradient-norm controls are separate from ordinary adaptation; unit-mean
sample weights alone are not treated as gradient matching.

## Reference results

CNN rows below are the archived three-seed camera-ready reference for **Table 2, TENT rows** (full deviations and seed values: Tables 26 / 27).
Accuracy is mean ± sample SD in percentage points; the other columns show means.
Fresh release checks are reported separately in [Reproducibility](docs/REPRODUCIBILITY.md) · [Code/paper semantics](docs/CODE_SEMANTICS.md).

| Dataset / backbone | Method | Accuracy (%) | MMC | ECE | NLL |
|---|---|---:|---:|---:|---:|
| CIFAR-100-C / ResNeXt AugMix | TENT | 67.84 ± 0.05 | 0.7461 | 0.0693 | 1.2547 |
| CIFAR-100-C / ResNeXt AugMix | TENT+MIRA | 68.69 ± 0.15 | 0.7305 | 0.0452 | 1.1940 |
| ImageNet-C / ResNet-50 V1 | TENT | 36.62 ± 0.12 | 0.4089 | 0.0478 | 3.2549 |
| ImageNet-C / ResNet-50 V1 | TENT+MIRA | 39.45 ± 0.14 | 0.4053 | 0.0361 | 3.0205 |

The selected **torchvision ViT-B/16 reference** uses LR 0.00025 / batch 64, with one measured seed:

| Method | Accuracy (%) | MMC | ECE | NLL |
|---|---:|---:|---:|---:|
| TENT | 41.64 | 0.5170 | 0.1167 | 3.0582 |
| TENT+MIRA | 44.76 | 0.4908 | 0.0549 | 2.8406 |

Paper Table 3 reports 42.03/45.17%. The new accuracies are 0.39/0.41 percentage points
lower; the gain is +3.12 points versus the paper's +3.14. Original checkpoint/seed/stream
provenance remains unconfirmed, so this is labeled a separate reconstruction with no seed SD.
Mean accuracy, ECE and NLL improve. Frost accuracy decreases by 2.66 points and brightness
ECE increases. No-feasible/fallback frequency is 99.62%; this result does not establish
jointly feasible scale selection on every batch. Full metrics and paired differences are
included in [code/results](code/results/). Previous timm references have separate profiles and archives.

<!-- Result figures reserved. Add the final plots here when ready. -->

## Standalone adapter

```python
from mira import MIRA, MIRAConfig
from mira.checkpoints import load_source_classifier

model, _ = load_source_classifier(
    "vit_b_16",
    "/path/to/vit_b_16-c867db91.pth",
)
adapter = MIRA(model.cuda(), MIRAConfig(lr=2.5e-4))
probabilities = adapter.predict_proba(images.cuda())  # unlabeled RGB batch in [0, 1]
adapter.reset()                                    # next corruption boundary
```

CNNs adapt normalization affine parameters. ViT adapts only LayerNorm weight and bias
(50 tensors / 38,400 parameters); other source weights remain fixed. Use the source model's
input normalization; the loader includes this ViT's mean (0.485, 0.456, 0.406) and
std (0.229, 0.224, 0.225). ImageNet-C tensors are already 224×224.

Defaults: α=0.03, ρ=0.90, τ=0.25, γ=1, K=300 and the original 16-radius grid, with
feature-wise He-in scaling. Gaussian draws are shared across candidate radii. Reliability
weights are detached; the preserved selector uses log-scale interpolation and its original
MI-first fallback. A fallback can bypass the agreement requirement; diagnostics report it.
Labels enter only metric evaluation, never scale selection or adaptation.

## Citation and attribution

```bibtex
@inproceedings{kim2026mira,
  title={MIRA: Mutual Information guided calibration for Reliable Test-Time Adaptation},
  author={Kim, Hyeongyu and Jang, Mingyeong and Song, Youngjun and Han, GeonHui and Hwang, Dosik},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026},
  url={https://openreview.net/forum?id=ujUYtJK5LG}
}
```

The implementation follows the [test-time-adaptation framework](https://github.com/mariodoebler/test-time-adaptation)
and RobustBench. Source notices are retained; see [licenses and attribution](docs/THIRD_PARTY.md).
The existing project webpage remains in `index.html` and `static/`; GitHub Pages serves the
root of `main`. README figure slots are reserved for final assets.

Configurations for the main CNN/ViT recipes live directly under `code/configs/`;
other controls are grouped by study in `code/configs/ablations/`. Large detailed logs
are packaged by study, with browsable seed/summary CSVs retained in `code/results/`.
