# Environment, data and checkpoints

## Reference environment

Use the Linux conda lock plus pip pins shown in the root README. The lock records the
actual reference conda packages; `requirements-reference.txt` records the Python packages
used by evaluation. The scientific core is frozen in `reference_manifest.json`.
PyTorch CUDA 12.1 requires a compatible NVIDIA driver. CPU tests do not require data or weights.

```bash
cd code
python scripts/check_environment.py
python scripts/validate_release.py
python -m unittest discover -s tests -v
```

Alternative installations may work, but changed timm, image decoding, CUDA or convolution
implementations can affect numerical results. CUDA reference runs use cuDNN benchmarking,
so exact bitwise equality is not promised. The original and fresh measurements are retained
separately, including differences in the final printed digits.

## Corruption data

Download the benchmark images from their official records:
[CIFAR-10-C](https://zenodo.org/records/2535967),
[CIFAR-100-C](https://zenodo.org/records/3555552), and
[ImageNet-C](https://zenodo.org/records/2235448).
The reference ImageNet-C assets contain 224×224 images. Preserve the official validation
image basenames and class directories: the frozen candidate list selects the target stream.
The loader preserves the archived shuffle/sample operations rather than selecting a new subset.

```text
corruption-data/
  CIFAR-10-C/
    labels.npy
    gaussian_noise.npy
    ...
  CIFAR-100-C/
    labels.npy
    gaussian_noise.npy
    ...
  ImageNet-C/
    gaussian_noise/5/n01440764/ILSVRC2012_val_00000293.JPEG
    ...
```

The 15-corruption order is gaussian_noise, shot_noise, impulse_noise, defocus_blur,
glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast,
elastic_transform, pixelate, jpeg_compression. Configs select severity 5 and preserve
that order. The CIFAR-10 objective controls use four corruptions and their own protocol;
the CIFAR-100 WRN construction controls use all severity-5 examples shuffled once.
Do not combine their metrics with the headline ResNeXt protocol.

## Exact source weights

Place each weight file under `MIRA_CHECKPOINT_ROOT` as follows. All full SHA256 identities
and byte sizes are in [reference_manifest.json](../code/reference_manifest.json).

| Relative path | Official source |
|---|---|
| `cifar100/corruptions/Hendrycks2020AugMix_ResNeXt.pt` | RobustBench GDrive ID `1ocnHbvDdOBLvgNr6K7vEYL08hUdkD1Rv` |
| `cifar100/corruptions/Hendrycks2020AugMix_WRN.pt` | RobustBench GDrive ID `1XpFFdCdU9LcDtcyNfo6_BV1RZHKKkBVE` |
| `cifar10/corruptions/Hendrycks2020AugMix_WRN.pt` | RobustBench GDrive ID `1wy7gSRsUZzCzj8QhmTbcnwmES_2kkNph` |
| `resnet50-0676ba61.pth` | [torchvision ResNet-50 V1](https://download.pytorch.org/models/resnet50-0676ba61.pth) |
| `resnet34-b627a593.pth` | [torchvision ResNet-34 V1](https://download.pytorch.org/models/resnet34-b627a593.pth) |
| `jx_vit_base_p16_224-80ecf9dd.pth` | [timm ViT-B/16 orig_in21k_ft_in1k](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth) |

CIFAR AugMix weights are the unmodified RobustBench checkpoint files, including any prefix
keys present in the originals. The loader handles those keys. A different model with the
same architecture name is not an interchangeable source checkpoint. No evaluation command
silently downloads a new checkpoint.

```bash
export MIRA_DATA_ROOT=/path/to/corruption-data
export MIRA_CHECKPOINT_ROOT=/path/to/source-checkpoints
export MIRA_RESULTS_ROOT=/path/to/results
export CUDA_VISIBLE_DEVICES=0
python scripts/check_assets.py --scope main --check-data  # two headline CNN weights
python scripts/check_assets.py --scope vit --check-data   # ViT only
python scripts/check_assets.py --scope all --check-data   # six weights and all datasets
```

## Commands, outputs and resume

```bash
bash scripts/run_main.sh
bash scripts/run_vit.sh
python scripts/run_reference.py --config configs/c100_full_s1.json
python scripts/run_reference.py --config configs/in50_tent_s1.json
python scripts/run_reference.py --config configs/vit_b16_full_s1.json
bash scripts/run_ablations.sh
python scripts/summarize.py --runs "$MIRA_RESULTS_ROOT" --output results_local
```

Use a separate output root for a smoke check:

```bash
MIRA_RESULTS_ROOT=/path/to/smoke-results \
  python scripts/run_reference.py --config configs/vit_b16_full_s1.json --max-batches 2
```

Smoke results are explicitly marked `smoke_only` and excluded from summaries.
Each complete evaluation writes its resolved config, parameter names, checkpoint hash,
code-source hashes, RNG seeds, ordered stream hashes, corruption metrics and batch
selector diagnostics. JSONL probes include candidate sigma, MI, agreement, interpolation
and fallback. CUDA allocated/reserved peaks and measured runtime have separate fields.

Every corruption begins from the source model and a fresh optimizer. Completion at a
corruption boundary is atomic. Rerunning the same config reuses completed work and the
saved stream/perturbation RNG states; an incomplete corruption is replayed from its source
state. Changed configs are rejected for old result IDs. Give a changed experiment a new run ID.

Stream RNG and perturbation RNG have separately recorded seeds. TENT/MIRA receive
corresponding target streams and batches. Sigma candidates reuse Gaussian draws within a
probe; stochastic loss draws follow the original implementation. The portable ordered-image
hash is based on data-root-relative identifiers, so it is independent of local installation paths.
The historical absolute-path stream hash is retained as an additional record.

Accuracy is a macro mean over corruptions in percent. MMC, NLL and 10-bin ECE are
computed within each corruption and macro averaged. Sample SD uses `ddof=1`; accuracy SD
is in percentage points. A single measured seed has blank SD, not zero SD or standard error.
