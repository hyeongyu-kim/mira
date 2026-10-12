# Code and paper semantics

Checked against `NeurIPS_2026_MIRA_camera_ready_v1.pdf`, Sections 4.1–4.3 and
Appendices D.3–D.6. The released Full TENT/MIRA recipe agrees with the core method.

| Paper operation | Executable behavior | Evidence |
|---|---|---|
| Cached last-layer Gaussian predictors, Section 4.1 | One backbone feature extraction; additive classifier logits are algebraically equivalent to sampling W/b | Explicit perturbed-classifier numerical test |
| Feature-wise He-in scale, D.4 | sqrt(2/D), batch mean absolute feature magnitudes, unit-mean factor clipped to [.25,4]; bias scale sqrt(2/D) | Independent factor test |
| Shared draws across candidate radii | One weight/bias Gaussian draw scaled by each radius; loss uses fresh draws | RNG capture and explicit-grid test |
| HE / EH / MI | Entropy of mean probability, mean sample entropy, HE−EH; no posterior fitting | Independent probability/entropy test |
| Agreement, Section 4.2 | MC-sample argmax agreement with the MC-mean argmax, averaged over K and B | Core selector inspected; no comparison to labels/MAP class |
| Selection and D.6 fallback | Earliest MI/agree feasible candidate; first MI-feasible otherwise; MI argmax fallback; conditional log-sigma interpolation | Feasible/fallback fixtures and source inspection |
| Detached MI-confidence gate, Section 4.3 | Centered exp(−MI/τ), unit mean, multiplied by unit-mean mean-prediction confidence, renormalized and detached | Literal weighted-loss and gradient test |
| HE+γEH | γ scales EH; public adapter and dedicated gamma sweep hold confidence exponent at 1 | Gamma and literal-gradient tests |
| Underlying TENT update | Normalization affine parameters only; SGD/Adam and reset per explicit protocol; main gradients unnormalized | Actual-gradient, reset and frozen-classifier tests |
| Metrics and streams | Pre-update MC prediction for MIRA; 10-bin ECE; matched stream/batches and separately recorded RNG seeds | Seed/config/stream audits |

Numerical implementation details supplement the symbolic equations: probabilities and
normalization denominators use small epsilon floors; feature factors use a finite-value/floor
guard; centered MI logits are clipped to [−20,20] before exponentiation. The exp-clamp can
change extreme relative weights, so the unclipped symbolic formula is not claimed to match
every possible input exactly. These existing guards are retained, not newly tuned for release.

The frozen legacy core's nondefault gamma can also enter its confidence exponent. The
headline reference has gamma=1, so it agrees with the paper. The public adapter and
`EH_coefficient_only` sensitivity profiles explicitly separate the EH coefficient from the
confidence exponent; the gamma sweep does not reuse that coupled legacy behavior.

Agreement is guaranteed only on jointly feasible selection samples; the existing fallback
does not enforce rho, and fresh loss draws need not preserve selection-sample agreement.
No target labels enter sigma selection, reliability weights or adaptation.
Performance benefits and gating advantages remain protocol dependent, as documented in
[results](RESULTS.md); semantic alignment is not a universal-improvement claim.

Recovered ViT settings identify ImageNet-C, learning rate 0.00025 and batch size 64,
but not source checkpoint, architecture library, seed or complete preprocessing/stream.
The selected release reference is the measured torchvision reconstruction (41.64/44.76%).
Previous timm references have separate versioned configs and archives. All three profiles
retain their source/checkpoint/protocol identities; exact historical Table3 reproduction is unconfirmed.

22 CPU tests pass, including four independent paper-equation checks and both transformer
implementations' feature capture, norm-only updates, classifier preservation and reset. Grouping configs and
archiving diagnostics leaves the frozen scientific core unchanged.
