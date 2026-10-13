# MIRA

**MIRA: Mutual Information guided calibration for Reliable Test-Time Adaptation**

Hyeongyu Kim, Mingyeong Jang, Youngjun Song, GeonHui Han, Dosik Hwang · NeurIPS 2026

Paper (coming soon) · [Project page](https://hyeongyu-kim.github.io/mira/) · [Documentation](docs/SETUP.md)

MIRA uses classifier perturbations to estimate prediction reliability and weight a
stochastic entropy objective for test-time adaptation.

## Supported experiments

**Methods:** TENT and TENT+MIRA.

| Dataset | Backbone | Batch size | Learning rate |
|---|---|---:|---:|
| CIFAR-100-C | ResNeXt AugMix | 64 | 0.001 |
| ImageNet-C | ResNet-50 | 64 | 0.00025 |
| ImageNet-C | torchvision ViT-B/16 | 64 | 0.00025 |

Additional experiments cover objective and gate ablations, perturbation construction,
and hyperparameter sensitivity. See [experiment details](docs/RESULTS.md).

<!-- Overview figure reserved. -->

## Installation

```bash
git clone https://github.com/hyeongyu-kim/mira.git
cd mira
conda create -n mira --file code/environment-conda-linux-64.lock.txt
conda activate mira
python -m pip install -r code/requirements-reference.txt
python -m pip install -e code --no-deps
cd code
```

Download the datasets and source checkpoints listed in [Setup](docs/SETUP.md),
then set the local paths:

```bash
export MIRA_DATA_ROOT=/path/to/corruption-data
export MIRA_CHECKPOINT_ROOT=/path/to/source-checkpoints
export MIRA_RESULTS_ROOT=/path/to/results
export CUDA_VISIBLE_DEVICES=0
```

## Run

Run these commands from `code/`:

```bash
bash scripts/run_main.sh        # CIFAR-100-C / ResNeXt and ImageNet-C / ResNet-50
bash scripts/run_vit.sh         # ImageNet-C / torchvision ViT-B/16
bash scripts/run_ablations.sh   # objective, gate, perturbation and sensitivity experiments
python scripts/summarize.py --runs "$MIRA_RESULTS_ROOT" --output results_local
```

Results, resolved configs and batch diagnostics are saved under `MIRA_RESULTS_ROOT`.
Rerunning a command reuses completed runs and resumes interrupted runs.
See [Setup](docs/SETUP.md) for individual commands and
[ViT profiles](docs/VIT_PROFILES.md) for alternative ViT configurations.

## Results

| Dataset / backbone | Method | Accuracy (%) | MMC | ECE | NLL |
|---|---|---:|---:|---:|---:|
| CIFAR-100-C / ResNeXt AugMix | TENT | 67.84 | 0.7461 | 0.0693 | 1.2547 |
| CIFAR-100-C / ResNeXt AugMix | TENT+MIRA | 68.69 | 0.7305 | 0.0452 | 1.1940 |
| ImageNet-C / ResNet-50 | TENT | 36.62 | 0.4089 | 0.0478 | 3.2549 |
| ImageNet-C / ResNet-50 | TENT+MIRA | 39.45 | 0.4053 | 0.0361 | 3.0205 |
| ImageNet-C / torchvision ViT-B/16 | TENT | 41.64 | 0.5170 | 0.1167 | 3.0582 |
| ImageNet-C / torchvision ViT-B/16 | TENT+MIRA | 44.76 | 0.4908 | 0.0549 | 2.8406 |

[Protocols and full results](docs/RESULTS.md) · [Reproducibility](docs/REPRODUCIBILITY.md) · [Result files](code/results/)

<!-- Result figures reserved. -->

## Citation

```bibtex
@inproceedings{kim2026mira,
  title={MIRA: Mutual Information guided calibration for Reliable Test-Time Adaptation},
  author={Kim, Hyeongyu and Jang, Mingyeong and Song, Youngjun and Han, GeonHui and Hwang, Dosik},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

The implementation builds on the [test-time-adaptation framework](https://github.com/mariodoebler/test-time-adaptation)
and RobustBench. See [licenses and attribution](docs/THIRD_PARTY.md).
