# Results and scientific protocols

## CNN camera-ready reference

These are the archived **three-seed TENT rows of Table 2 / 26**, in
`NeurIPS_2026_MIRA_camera_ready_v1.pdf`, pages 8 / 24; Table 27 (page 25) lists the corresponding seed values. Each entry is mean ± sample SD (ddof=1).
Accuracy and its SD use percent and percentage points. The reference seed CSV is kept
separate from the fresh release-validation CSV.

| Dataset / source | Method | Accuracy (%) | MMC | ECE | NLL |
|---|---|---:|---:|---:|---:|
| CIFAR-100-C / ResNeXt AugMix | TENT | 67.84 ± 0.05 | 0.7461 ± 0.0007 | 0.0693 ± 0.0004 | 1.2547 ± 0.0021 |
| CIFAR-100-C / ResNeXt AugMix | TENT+MIRA | 68.69 ± 0.15 | 0.7305 ± 0.0012 | 0.0452 ± 0.0007 | 1.1940 ± 0.0111 |
| ImageNet-C / ResNet-50 V1 | TENT | 36.62 ± 0.12 | 0.4089 ± 0.0014 | 0.0478 ± 0.0018 | 3.2549 ± 0.0075 |
| ImageNet-C / ResNet-50 V1 | TENT+MIRA | 39.45 ± 0.14 | 0.4053 ± 0.0014 | 0.0361 ± 0.0022 | 3.0205 ± 0.0020 |

CIFAR-100-C uses the ResNeXt AugMix source, SGD at 0.001 and batch 64. ImageNet-C uses
ResNet-50 V1, SGD at 0.00025 and batch 64. Both use momentum 0.9, Nesterov, zero weight
decay, one update per batch, 15 severity-5 corruptions, 10,000 samples per corruption and
source/optimizer reset between corruptions. This public ResNeXt recipe is the camera-ready
TENT learning-rate protocol; older submission/rebuttal protocols must not be pooled with it. The additional Table 28
TENT rows use seeds 2/4/6; they are a separate retained comparison, outside this default
profile and its seeds 1/2/3. No exact Table 28 replay command is claimed here.

## Selected ViT batch-64 reference

These are **newly measured seed-1 results** selected for the code release. No multi-seed
uncertainty is implied:

| Method | Accuracy (%) | MMC | ECE | NLL |
|---|---:|---:|---:|---:|
| TENT | 45.96 | 0.7330 | 0.2736 | 4.0038 |
| TENT+MIRA | 50.02 | 0.6393 | 0.1410 | 2.9037 |

The source is timm `vit_base_patch16_224.orig_in21k_ft_in1k`, checkpoint
`jx_vit_base_p16_224-80ecf9dd.pth`. SGD is 0.001 with momentum 0.9, Nesterov and zero
weight decay. The evaluation uses batch 64, one update per batch, 15 severity-5 corruptions,
10,000 samples each and reset per corruption. Input mean/std is (0.5,0.5,0.5).
Only LayerNorm weight and bias adapt (50 tensors, 38,400 parameters).

MIRA uses α=.03, ρ=.90, τ=.25, γ=1, K=300, feature-wise He-in perturbations and the
original 16-radius grid. No hyperparameter was selected from target labels. The paired
changes are +4.0613 accuracy percentage points, −0.132639 ECE and −1.100149 NLL.
Accuracy improves in 11/15 corruptions; ECE and NLL improve in 14/15. Impulse-noise accuracy
decreases by 17.77 points, so the mean result must not be described as improvement on every
corruption. No-feasible/fallback frequency is 94.862%; selected sigma mean is 0.9661,
median 1, and range [0.5,2]. A low ECE by itself does not establish better prediction quality.

Historical paper Table 3, page 9, reports 42.03/45.17% accuracy; the original complete
checkpoint/config/log provenance was not recovered. Those rows are retained only as an
explicitly unresolved historical comparison in `vit_vs_paper_table3.csv`. The selected
batch-64 reference is not claimed to reproduce those numbers. The previous batch-128
three-seed validation is in `results/archive/vit_batch128/`, with a separate executable profile.

## Supported appendix comparisons

The following mappings use the current camera-ready PDF's numbering. Every listed table
has archived seed evidence and executable portable configs. Numerical values and sample
SDs were independently recomputed from seed CSVs; printed cells match the PDF.

| Table | Page | Released comparison | Scientific distinction |
|---|---:|---|---|
| 1 / 6 | 8 / 14 | ResNet-34 objective/gate ablation | SGD 0.00025; ordinary gradients; Table 6 adds confidence-only |
| 2 / 26 / 27, TENT rows | 8 / 24 / 25 | ResNeXt and ResNet-50 main comparison | Separate dataset-specific learning rates |
| 7 | 14 | Global gradient-matched objective/gates | Actual adaptation-gradient global L2 norm targeted to 1; separate controls |
| 8 | 15 | Ordinary CIFAR-10 WRN objective/gates | Adam 0.001; four corruptions; K=100 |
| 9 | 15 | Feature/scalar/class scale | CIFAR-100 WRN; K=100; shuffle only; fixed He-in base |
| 10 | 15 | He-in/fan-in/absolute/weight magnitude | Same WRN protocol; feature-wise scale fixed |
| 13 | 17 | Joint α–ρ sweep | Original 16-radius grid; K=300; two headline protocols |
| 14 | 17 | γ sweep | HE+γ·EH; confidence exponent stays 1; ordinary gradient update |
| 15 / 16 | 18 | Grid sensitivity and diagnostics/cost | Fixed K=300; grid-dependent normalized MI retained |

CIFAR-10 controls use Gaussian noise, snow, fog and pixelation. The WRN construction
protocol uses all 10,000 severity-5 images per corruption, shuffled once; the ResNeXt
headline loader also performs a sampling operation. These streams remain separate.
For Table 7, unit-mean sample weights are followed by a detached reciprocal of the complete
adaptation-parameter global L2 gradient norm before the optimizer step. Maximum recorded
norm error in 45 controls is 0.000248432; matching sample-weight means alone is insufficient.

The sweeps support a common practical setting, not a universal optimum. Confidence-only
can outperform Full on ResNet-34, and global norm matching reverses the Full-versus-unweighted
accuracy difference on ImageNet-C. γ=2 improves ImageNet-C accuracy/NLL while worsening
ECE. High fallback frequency limits conclusions about the agreement guard. Coarsening the
grid harms ImageNet-C accuracy/calibration even though endpoint-selection frequencies are zero.
These qualifications are consistent with the revised camera-ready discussion.

The original sigma grid is
[1e−5,1e−4,3e−4,1e−3,3e−3,1e−2,3e−2,5e−2,1e−1,3e−1,5e−1,1,2,3,4,5].
The selector threshold is MI_min+α·(MI_max−MI_min); changing the grid changes this range.
If no candidate satisfies both MI and agreement, the original selector takes the first MI-feasible
candidate, with an MI-maximum fallback if necessary. Log-sigma interpolation requires the
neighboring MI crossing and agreement at both endpoints. No new fallback is introduced.
Lower/upper boundary diagnostics include fractional indices ≤1 / ≥L−2, respectively,
covering the endpoints and first/last interpolation intervals. Grid and K sweeps are distinct.

## Files

All paths below are relative to [code/results](../code/results/):

- `seed_results.csv`, `mean_sd_summary.csv`, `paired_seed_differences.csv`: 217 archived CNN runs.
- `diagnostics/cnn_reference.tar.gz`: per-run portable config/provenance, corruption metrics, compressed batch logs,
  matched-stream and actual-gradient audits.
- `release_validation_*.csv`, `diagnostics/cnn_release_checks.tar.gz`: fresh CNN checks, kept separate from the archive.
- `vit_seed_results.csv`, `vit_mean_sd_summary.csv`, `vit_results.tex`: current batch-64 seed-1 reference.
- `vit_paired_*differences.csv`, `diagnostics/vit_batch64.tar.gz`: paired differences and complete current ViT diagnostics.
- `archive/vit_batch128/` and `diagnostics/vit_batch128.tar.gz`: the previous measured ViT profile, never pooled with batch 64.

The explicitly labeled `c100_published_table23_s1` row is an older single-run unweighted HE+EH,
ρ=.95 historical control. Its separate scientific group is retained for traceability and is
not the current Full MIRA headline reference or a three-seed result.

[paper_numeric_audit.csv](paper_numeric_audit.csv) enumerates the printed-cell comparisons;
[paper_numeric_audit.json](paper_numeric_audit.json) records the source PDF SHA256 and scope.
[code/docs/table_commands.json](../code/docs/table_commands.json) provides the exact configs.
Only TENT/MIRA and their released controls are covered; other paper plug-in comparisons,
CLIP, historical deeper-perturbation/K studies and figure-specific streams are not advertised
as supported reproduction commands. README figures are reserved for later assets.
