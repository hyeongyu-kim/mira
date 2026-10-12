# Archived CNN camera-ready evidence

217 seed runs underpin the supported CNN objective, gating, construction and sensitivity
profiles. `index.csv` locates a portable result/provenance record, per-corruption CSV and
compressed batch diagnostics for each run. The configs are the exported public configs;
metric values and original result SHA256 identities are retained. Absolute local asset paths
are omitted. Early runs lacking a worker hash are listed in `reproducibility_audit.json`.

The original target-stream hashes are tied to their historical paths for ImageNet images;
new release runs also record a path-independent ordered-image identifier hash. Batch
summaries include sigma, MI, agreement, interpolation, feasibility/fallback and gradient norm.
Complete candidate-grid arrays for new ViT and fresh release checks are in their separately
labeled validation directories. Adaptation states, weights and images are not distributed.
