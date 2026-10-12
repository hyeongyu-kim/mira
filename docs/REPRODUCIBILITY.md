# Reproducibility coverage

The code release supports **TENT/MIRA on CNNs and ViT-B/16**, including their explicit
CNN controls. The selected ViT reference is torchvision, LR0.00025 / batch64 / seed1.
Previous timm profiles retain separate configs and evidence. Scope, archived evidence and
fresh execution are recorded separately; the release does not claim every experiment in the
paper has been freshly rerun.

## Paper correspondence

The checked document is `NeurIPS_2026_MIRA_camera_ready_v1.pdf` (SHA256
`63602757825d66027b7b496c40ec0319006fb4212091e58590c3737a392d029b`).
**884 displayed numerical cells** match their archived seed evidence at printed precision:
Table 1 (means and SDs), Table 2's four released TENT/MIRA rows, Tables 6–10 and 13–16,
Table 26's released pairs and Table 27's 12 individual seed rows.
Statistics were recomputed from 217 seed rows across 73 separate scientific groups, with
sample SD (ddof=1). Table 1 uses accuracy fractions; the other supported tables use percent.
The audit includes the Table 16 memory-unit conversions and zero boundary frequencies.

[Cell-by-cell CSV](paper_numeric_audit.csv) · [Audit scope and PDF hash](paper_numeric_audit.json) ·
[Table-to-config mapping](../code/docs/table_commands.json).

The 217 archived runs have no recorded integrity, batch-count, selection, stream-group or
recorded-RNG-group failures. The 45 global gradient-norm controls have maximum absolute
norm error 0.000248432. The RNG audit uses the first batch of each corruption where a CUDA
state hash was actually saved; it does not invent hashes for earlier runs. Six early runs lack
a recorded worker hash; their configs/checkpoints/metrics are retained and this limit is listed
in `cnn_reference/reproducibility_audit.json` inside `results/diagnostics/cnn_reference.tar.gz`.

## Fresh full checks

Thirteen complete CNN checks are retained: CIFAR-100-C ResNeXt TENT/MIRA seeds1/2/3;
ImageNet-C ResNet-50 TENT/MIRA seeds1/2/3; and representative ResNet-34 Full seed1.
Each covers 15 corruptions, 150,000 images and 2,355 batches. The two current ViT checks
are additional full seed1 evaluations, with the same image/batch counts.
The main CNN runs have 90 matched method/seed/corruption comparisons; ordered target
streams, sample counts and batch counts agree in all of them. Current ViT has 15 additional
matched corruption streams. The source checkpoint is reloaded for each run and source
model/optimizer reset is verified at each corruption boundary.

Fresh and archived CNN accuracy are shown separately:

| Dataset / backbone | Method | Fresh accuracy (%) | Archived accuracy (%) |
|---|---|---:|---:|
| CIFAR-100-C / ResNeXt | TENT | 67.85 ± 0.05 | 67.84 ± 0.05 |
| CIFAR-100-C / ResNeXt | TENT+MIRA | 68.70 ± 0.16 | 68.69 ± 0.15 |
| ImageNet-C / ResNet-50 | TENT | 36.62 ± 0.10 | 36.62 ± 0.12 |
| ImageNet-C / ResNet-50 | TENT+MIRA | 39.45 ± 0.14 | 39.45 ± 0.14 |

Maximum absolute seedwise accuracy difference is 0.012 pp on CIFAR-100-C, 0.018667 pp
on ImageNet-C ResNet-50, and 0.026001 pp for the representative ResNet-34 run. Some final
printed digits and SDs differ. The source environment uses cuDNN benchmarking, so these
checks establish the same observed performance range rather than bitwise identity.
Fresh paired results retain accuracy/ECE/NLL improvements in the two main CNN comparisons.
See `release_validation_vs_archived.csv` for all four metric differences; no claim of
statistical significance is inferred from three repeats.

The selected torchvision ViT measurements are 41.64/.5170/.1167/3.0582 for TENT and
44.76/.4908/.0549/2.8406 for TENT+MIRA (Accuracy %, MMC, ECE, NLL). Seeds2/3 are
prepared but unmeasured; SD fields are blank. Full execution used GPU0 and preparation commit
`7143945dbc1124bf69b8784a4f4fb3dff29f0a78`; the worker and frozen core are unchanged.
The recovered LR/batch agree with historical notes, but original checkpoint/seed/stream
provenance remains unconfirmed. Table3 reports 42.03/45.17: differences are −.3927/−.4127 pp,
while the gain +3.12 pp is close to +3.14 pp. Exact original reproduction is not claimed.
Fallback frequency is 99.62%; improved mean metrics do not prove universal joint feasibility.
Previous timm batch64 and batch128 results are archived as separate versioned protocols.

## Implementation checks

22 CPU tests pass, covering scientific objective equivalence, independent paper equations, shared Gaussian probing,
classifier restoration, detached sample weights, exact model/optimizer reset, actual global
adaptation-gradient norm matching, timm/torchvision classifier-input caching, LayerNorm-only ViT adaptation,
and safe switching between versioned profiles without reusing old result IDs.
Six earlier GPU smokes cover ResNeXt, ResNet-50 and the preserved timm ViT paths.
Two new torchvision TENT/MIRA smokes use the default public command and match all four
original two-batch metrics within 1e−6; they are explicitly excluded from benchmark rows.
See [selected-profile validation](../code/docs/validation_checks/torchvision_profile_validation.json)
and [full-run evidence](../code/docs/validation_checks/torchvision_vit_full_validation.json),
with [CPU test output](../code/docs/validation_checks/torchvision_cpu_tests.txt).
A prior actual-weight timm ViT standalone-API check also matches reference outputs and updated
state tensors exactly, with unchanged classifier and exact reset.

The MIRA scientific core is byte-identical to the frozen experimental snapshot:
`914e51046485bbf177ae96a78911766ed53054b4e328eb44f6801afc253b21b7`.
The source snapshot's original commit is `8b2ed63255d906b6b0afaefc6ae630b9ec4bdaa1`,
with modified worktree files identified by per-file hashes. Portable model dispatch, profile
selection, integrity/resume checks, provenance metadata and packaging have separate hashes.
The final worker records the public repository commit, original source commit and source manifest.
Completed result reuse checks the source/checkpoint and scientific config first, including
actual gradient-norm target. Completed matching runs were reused successfully; changed learning
rate and gradient-norm target were both rejected for an existing result ID. Public package
tests and commands are checked again after a local clone.

```bash
cd code
python scripts/validate_release.py
python -m unittest discover -s tests -v
python scripts/check_environment.py
python scripts/check_assets.py --scope all --check-data
bash scripts/run_vit.sh --dry-run
```

The clean-clone validation passed on commit
`cab51ed5c54b8ceb7ea60c39b084f8b0854f585b`: an isolated editable install, all source/config
hashes and assets, 15 CPU tests, and two final-worker ViT GPU smokes. Both smokes match
all four original batch64 smoke metrics within 1e−6 and correctly record that public commit
with a clean worktree. That clone check covers the preserved timm profile and common runtime.
The subsequent torchvision promotion changes the default configs and evidence while retaining
the same frozen scientific files; new default-command GPU smokes validate its portable paths.
The conda environment was inherited
from the measured reference packages for this installation check; a from-scratch conda
network installation was not performed. See
[clone_validation.json](../code/docs/validation_checks/clone_validation.json).

The checked conda/Python versions and JPEG decoder match the reference environment.
CNN/timm checks used NVIDIA RTX A6000 physical GPUs2/3; the selected torchvision pair used GPU0.
Runtime depends on host load
and allocator warm-up; Table16's archived cost records are not promised as identical fresh
timings. CUDA allocated and reserved peaks remain separate. The conda lock preserves the
reference JPEG backend, which can differ from a pip Pillow wheel despite the same version.

Batch exports retain selection, interpolation, normalized-MI range, candidate curves,
feasibility/fallback, agreement, gradient norms and recorded RNG hashes. Redundant
per-feature scaling vectors are omitted from distributed JSONL exports to keep repository
size manageable; original full logs remain in the authors' experiment archive and the
runner retains its original diagnostic output. This affects exports only, not computation.
The existing absolute-path stream hash is preserved; new measurements also record
path-independent data-root-relative ordered-image hashes.

## Coverage limits

EATA/SAR/ROID plug-ins and CLIP are outside this public code release. Historical deeper
perturbation, K-budget, cross-architecture figure diagnostics and figure-specific streams
are not included as verified public reproduction commands. No manuscript or Overleaf edit
is made by these preparation scripts. README figure slots are empty; existing project-page
assets are retained. GPU experiments are not run by GitHub Actions; its workflow performs
CPU invariants, source/config/statistics checks and profile command selection.
