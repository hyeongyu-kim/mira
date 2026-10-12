# Third-party source notices

The reference runtime is derived from the test-time-adaptation framework by Mario Döbler
and Robert Marsden. Its original MIT copyright and license are retained in `LICENSE`.
The embedded RobustBench code retains its source notices and the MIT notice in
`licenses/robustbench_LICENSE`.

Method files retain attribution to TENT and its upstream repository. Source and doubled-entropy
controls support the MIRA ablations. EATA/SAR/ROID integrations and the CLIP model factory are
excluded from this public candidate. The standalone adapter depends on PyTorch and the
preserved MIRA core; the source-model loader also uses torchvision/timm. The ImageNormalizer
in that loader follows the retained RobustBench normalization helper under its MIT notice.
Installed dependencies such as PyTorch, torchvision and timm keep their own licenses.
The repository does not redistribute pretrained checkpoints or benchmark images.

The Gaussian probing code is the authors' local experimental implementation, preserved
byte-for-byte as recorded in `reference_manifest.json`; the surrounding public API and portable
entrypoints were prepared for the camera-ready reproducibility audit.
