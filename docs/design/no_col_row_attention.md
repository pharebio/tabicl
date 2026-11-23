# Removing column-wise and row-wise attention

Scope for the `no-col-row-attention` branch. Branched from `ppb-inference-benchmark`
at 8c6d7e9, so the PPB performance and library-inference harnesses are available for
before/after comparison. No model code has been written yet; this file defines what
would be built and how it would be judged.

## 1. What the three stages cost

TabICL runs three stages in sequence: column-wise embedding (`ColEmbedding`), row-wise
interaction (`RowInteraction`), and dataset-wise in-context learning (`ICLearning`).

Parameter counts, from `tabicl-regressor-v2-20260212.ckpt`:

| stage | parameters | share |
|---|---|---|
| `col_embedder` | 0.875 M | 3.1% |
| `row_interactor` | 0.397 M | 1.4% |
| `icl_predictor` | 27.273 M | 95.5% |
| total | 28.545 M | |

Wall-clock per forward pass, measured on one RTX 6000 Ada at the library-inference
shape (1 table, 5,626 context rows, 50,000 test rows, fp32 inputs, the shipped default
`InferenceConfig` so AMP and its memory-derived batch sizing are both on, median of 3
after warm-up). Reproduced by
`benchmarks/stage_timing/measure_stage_timing.py`, which writes `stage_timing.csv`
beside itself:

| H | col | row | icl | total | col+row share | peak GPU |
|---|---|---|---|---|---|---|
| 23 | 0.076 s | 0.147 s | 0.105 s | 0.328 s | 68.0% | 4.81 GB |
| 62 | 0.201 s | 0.300 s | 0.103 s | 0.604 s | 82.9% | 11.57 GB |
| 100 | 0.310 s | 0.457 s | 0.111 s | 0.878 s | 87.4% | 18.15 GB |
| 200 | 0.726 s | 1.084 s | 0.128 s | 1.938 s | 93.4% | 30.09 GB |
| 720 | 2.577 s | 3.376 s | 0.129 s | 6.081 s | 97.9% | 45.19 GB |

Two facts follow, and together they are the case for the branch:

- **The ICL stage does not depend on feature width.** It holds 95.5% of the parameters
  and takes 0.103–0.129 s at every width from 23 to 720. All growth in feature width is
  in the two stages that hold 4.5% of the parameters.
- **Those two stages grow about linearly in width**, at roughly 8.3 ms per column
  (linear fit over H >= 62; residuals within ±10%, so the fit is a description of the
  measured range, not a law).

This is consistent with the benchmark figure already reported in
`benchmarks/ppb_performance/report.md` — a model stage linear in feature width up to
100 columns — and locates the H-independent intercept in that fit as the ICL stage.

### The activation expansion

`ColEmbedding` turns every cell into a 128-d vector, so the tensor handed to
`RowInteraction` has shape `(B, T, H+4, 128)`. At H=62 and T=55,626 that is 1.9 GB in
fp32 against the table's own 14 MB — a 128-fold per-cell expansion that then gets
collapsed back to 512 numbers per row. The peak-memory column above is mostly this
tensor and the row stage's attention intermediates over it.

A per-cell expansion of some size is hard to avoid; what the attention stages add is
that the expanded tensor must be *attended over*, by inducing points across rows in the
column stage and across the H+4 tokens of a row in the row stage. Removing both stages
does not remove the expansion so much as make it streamable: the cells can be embedded
and reduced chunk by chunk, so peak memory becomes a chosen chunk size rather than a
function of the table. Section 4 measures what that is worth.

Neither stage can be removed alone and get this: dropping only the row stage still
builds the full tensor in the column stage, and dropping only the column stage still
builds one (from `in_linear`) for the row stage to attend over.

## 2. The interface the ICL stage requires

Any replacement encoder has to produce what `ICLearning` already expects, because the
point of the branch is to keep the 27.3 M pretrained ICL parameters.

- **Shape** `(B, T, 512)`, where 512 = `embed_dim` (128) x `row_num_cls` (4).
- **Distribution.** `RowInteraction` applies `out_ln = LayerNorm(128)` to each of the 4
  CLS-token outputs and then flattens (`interaction.py`, `_aggregate_embeddings`). So
  each 128-wide chunk of the 512 arrives approximately zero-mean and unit-variance,
  times a learned affine. A replacement should match that, not merely the shape.
- **Targets enter later.** `ICLearning._icl_predictions` adds a y-embedding to the
  context rows of R itself. The encoder's output must carry feature information only.
  Note that the pretrained column stage is target-aware (`col_target_aware=True`), so
  the context rows' representations already depend on y before this point.
- **Row roles.** Test rows attend to context rows only, via the `train_size` argument.
  The encoder sees both and must not mix them: context statistics have to be computed
  from context rows alone or test labels leak.

## 3. What "removing" can mean

Four variants, in increasing order of how much they change. The saving column is a
ceiling -- it credits the removed stage with zero replacement cost -- and is computed
from the H=62 row above:

| variant | removes | keeps | time saved at H=62 | expansion becomes streamable |
|---|---|---|---|---|
| B | row attention | column embedding | ~49% | no |
| C | column attention | row attention | ~33% | no |
| D | both | ICL only | ~82% | yes |
| D' | both, plus ICL finetune | nothing frozen | ~82% | yes |

**Variant D is the one worth building.** B and C each buy about half the time and
neither lifts the memory ceiling. D is also the cleanest to state: raw preprocessed
features to `(B, T, 512)` with no attention anywhere before the ICL stage.

## 4. What the replacement encoder has to preserve

The column stage is not only expensive, it does a specific job: a set transformer over
rows gives each column an embedding conditioned on that column's own distribution and
on y. That is what lets one model read tables whose columns it has never seen, and it
is why column order can be permuted to build the 8-member ensemble.

A naive replacement — a shared per-cell MLP, mean-pooled over columns — throws this
away. It treats every column identically, so the row representation becomes a bag of
values with no column identity, and nearly all signal is lost. Two ways to keep column
identity without attention:

- **Per-column statistics as the column's code.** Compute cheap closed-form summaries
  of each column from the context rows (mean, sd, quartiles, covariance with y), map
  them through a small MLP to a 128-d column code, concatenate that code to each cell
  value, and sum or mean over columns. This uses the same information the set
  transformer uses — the column's distribution conditioned on y — computed in closed
  form instead of with 3 ISAB blocks. It keeps variable width and column-order
  invariance.
- **A learned per-column embedding table.** Simpler and strictly stronger for a
  fixed-width application, but it fixes H at training time and so gives up the
  foundation-model property. Reasonable as a control, not as the deliverable.

The first is the candidate to build. The second is worth training alongside it as an
upper bound on what a fixed-width encoder can do.

### What the candidate costs

A 0.381 M-parameter sketch of the statistics encoder — 6 context statistics per column
through an MLP to a 128-d code, concatenated to each cell value, a 2-layer cell MLP,
mean over columns, then an MLP to 512 and a LayerNorm — measured at the same shape and
on the same GPU, with the column code computed once per fold and rows streamed in
chunks of 16,384:

| H | replacement | peak GPU | pretrained col+row | stage speedup | model total before | after | overall |
|---|---|---|---|---|---|---|---|
| 23 | 0.018 s | 0.78 GB | 0.223 s | 12.4x | 0.328 s | 0.123 s | 2.7x |
| 62 | 0.033 s | 1.77 GB | 0.501 s | 15.2x | 0.604 s | 0.136 s | 4.4x |
| 100 | 0.053 s | 2.74 GB | 0.768 s | 14.5x | 0.878 s | 0.164 s | 5.4x |
| 200 | 0.104 s | 5.29 GB | 1.810 s | 17.4x | 1.938 s | 0.231 s | 8.4x |
| 720 | 0.363 s | 18.52 GB | 5.952 s | 16.4x | 6.081 s | 0.492 s | 12.4x |

The `after` column pairs the replacement with the ICL time measured for that width in
the table above, so both halves come from the same run. The ICL stage on its own peaks
at 1.13 GB against 11.57 GB for the full pipeline at H=62, so the new memory floor is
set by the encoder's chunk and not by the ICL stage. The replacement's own peak still
grows with H at a fixed chunk, but unlike the attention stages it shrinks freely with
the chunk, because nothing attends over the per-cell tensor before it is reduced.

These are the ceiling, not a result: an untrained encoder of the right shape. They say
what the speedup would be if a trained encoder of this size reaches the ICL stage's
input manifold, which is exactly what phase 1 of section 5 tests. The architecture
being 15x faster is not evidence that it works.

### Consequence for the ensemble

`TabICLRegressor` builds its 8 members from feature shuffles crossed with
normalization methods (`EnsembleGenerator`, `norm_methods=["none","power"]`). A
column-order-invariant encoder makes the feature shuffles a no-op, leaving 2 distinct
members. That is a further 4x inference saving, and a loss of whatever variance
reduction the shuffles were buying. The benchmark must therefore compare
like-for-like: either report the new model at `n_estimators=2` against the old at 8
(the honest end-to-end comparison), or hold the ensemble fixed and compare per-member.
Both numbers should appear.

## 5. Training

The infrastructure needed already exists and does not have to be built:

- `tabicl.train` pretrains from the synthetic prior, on the fly or from a prior
  directory, with `--freeze_col`, `--freeze_row`, `--freeze_icl`
  (`train/_train_config.py:286,300,311`).
- `tabicl._finetune` finetunes with the same three flags
  (`_finetune/base.py:288-290`), and `tests/test_finetune_freezing.py` already covers
  every combination.

Two phases, cheapest first:

1. **Distillation in representation space.** Freeze the whole pretrained model, run
   prior tables through `col_embedder` + `row_interactor` to get target R, and train
   the replacement encoder to regress onto it under L2 plus a cosine term. No ICL
   forward pass, no labels, no quantile loss. This is the cheap experiment that says
   whether a non-attentive encoder can reach the ICL stage's input manifold at all,
   and it should be run before anything else.
2. **End-to-end with ICL frozen** (`--freeze_icl true`), on the prior, against the
   real quantile loss. Unfreezing the ICL stage (variant D') is a fallback if the
   frozen ICL stage turns out to be unreachable; it costs the ability to claim the
   pretrained ICL weights are being reused unchanged.

## 6. Code changes required

In `tabicl`:

- A new `src/tabicl/_model/row_encoder.py` holding the replacement, with the same
  `forward(X, y_train, ...)` and `InferenceManager` batching contract the other two
  stages use.
- `TabICL.__init__` gains a flag selecting the encoder (default off, so the
  pretrained path is untouched) and `_train_forward` / `_inference_forward` branch on
  it. `InferenceConfig` gains a config for the new stage or reuses `ROW_CONFIG`.
- A training entry point and a checkpoint carrying the new config keys.
  `_sklearn/regressor.py::_load_model` already accepts a local `model_path`, so a
  locally trained checkpoint needs no loader changes.

In `proppredict` (branch `feature/tabicl-inference`):

- `models/tabicl_model.py::checkpoint_info` always calls
  `hf_hub_download(repo_id=HF_REPO_ID, ...)` and has no local-file branch, and
  `fit_fold` checks the resulting sha256 against the recorded run. Evaluating a
  locally trained checkpoint needs `checkpoint_version` to accept an existing path,
  still hashing and recording it. This is small and is the only change the harness
  needs.

## 7. Evaluation

- **Representation fidelity.** Distillation MSE and per-chunk cosine against the
  pretrained R, on held-out prior tables. Reported before any downstream number.
- **PPB performance.** Re-run the 5x5 folds for `tabicl_monroe_pca62` and
  `tabicl_monroe` and compare R2, MAE and Spearman against
  `benchmarks/ppb_performance/metrics_table.csv`, through the same Westfall-Young maxT
  bootstrap over test compounds already used there.
- **Inference speed.** Re-run `benchmarks/ppb_inference` for a new h/1M row.
- **Regression tests.** `tests/test_prediction_consistency.py`, `tests/test_inference.py`
  and `tests/test_inference_with_cache.py` must still pass with the flag off, which is
  what keeps the default path honest.

**Prerequisite, and it is not optional.** TabICL derives inference batch sizes from GPU
memory free at predict time, and varying batch size alone moves individual predictions
by up to 1.4e-02 on a target of sd ~0.6 (measured under fixed code, safety_factor 0.9
vs 0.05). Any before/after comparison must pin `inference_config` explicitly on both
sides, or that drift will be read as an effect of the architecture change.

## 8. Out of scope

The ICL stage's own architecture; the synthetic prior; the preprocessing and
normalization pipeline; the classification path, finetuning and forecasting entry
points; and anything that changes the numbers already reported in
`benchmarks/ppb_performance/report.md`.
