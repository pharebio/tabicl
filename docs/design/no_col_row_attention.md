# Removing column-wise and row-wise attention

Scope for the `no-col-row-attention` branch, branched from `ppb-inference-benchmark` at
8c6d7e9 so the PPB performance and library-inference harnesses are available for
before/after comparison.

The branch implements **rung 4** of the ladder in section 3: both attention stages
deleted, a row vector projected straight into the pretrained in-context-learning stage.
That is the architecture molecular ICL models assume when they feed a fixed-length
encoder output into an ICL transformer, and the branch exists to price that assumption
rather than to adopt it.

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
after warm-up). Reproduced by `benchmarks/stage_timing/measure_stage_timing.py`, which
writes `stage_timing.csv` beside itself:

| H | col | row | icl | total | col+row share | peak GPU |
|---|---|---|---|---|---|---|
| 23 | 0.076 s | 0.147 s | 0.105 s | 0.328 s | 68.0% | 4.81 GB |
| 62 | 0.201 s | 0.300 s | 0.103 s | 0.604 s | 82.9% | 11.57 GB |
| 100 | 0.310 s | 0.457 s | 0.111 s | 0.878 s | 87.4% | 18.15 GB |
| 200 | 0.726 s | 1.084 s | 0.128 s | 1.938 s | 93.4% | 30.09 GB |
| 720 | 2.577 s | 3.376 s | 0.129 s | 6.081 s | 97.9% | 45.19 GB |

Two facts follow, and together they are the case for the branch:

- **The ICL stage does not depend on feature width.** It holds 95.5% of the parameters
  and takes 0.103-0.129 s at every width from 23 to 720. All growth in feature width is
  in the two stages that hold 4.5% of the parameters.
- **Those two stages grow about linearly in width**, at roughly 8.3 ms per column
  (linear fit over H >= 62; residuals within +/-10%, so the fit describes the measured
  range rather than stating a law).

This is consistent with the model stage reported linear in feature width up to 100
columns in `benchmarks/ppb_performance/report.md`, and locates the H-independent
intercept of that fit as the ICL stage.

### Which axis each stage attends over

Worth stating explicitly, because the names invite the opposite reading and the
distinction decides what an ablation is testing. `ColEmbedding`'s first act is
`features = X.transpose(1, 2).unsqueeze(-1)  # (B, H+C, T, 1)` (`embedding.py:489`),
after which the set transformer runs over **T**, with each column a separate sequence.

| stage | attends across | within |
|---|---|---|
| `ColEmbedding` | rows | a column |
| `RowInteraction` | features | a row |
| `ICLearning` | context examples | the dataset |

So "column-wise embedding" means *an embedding of each column, produced by attending
over rows* -- not attention over columns. Attention over features is `RowInteraction`.
Attention over context examples, the ICL mechanism proper, is a separate third stage
that already exists and that this branch leaves untouched.

A related correction on complexity: TabICL is not a full 2D-attention model. The column
stage is induced self-attention with 128 inducing points and context-only keys
(`col_num_inds=128`, `layers.py:648`), so it is **linear in rows**, not quadratic; and
every ICL query attends to context rows only (`layers.py:437`,
`k = v = q[..., :train_size, :]`), so that stage is `O(n * n_ctx * d)`, not `O(n^2 * d)`.
The `O(n*f^2 + n^2*f)` cost of a full factorized 2D transformer does not describe this
model.

### The activation expansion

`ColEmbedding` turns every cell into a 128-d vector, so the tensor handed to
`RowInteraction` has shape `(B, T, H+4, 128)`. At H=62 and T=55,626 that is 1.9 GB in
fp32 against the table's own 14 MB -- a 128-fold per-cell expansion that then gets
collapsed back to 512 numbers per row. The peak-memory column above is mostly this
tensor and the row stage's attention intermediates over it.

A per-cell expansion of some size is hard to avoid; what the attention stages add is
that the expanded tensor must be *attended over*. Removing both stages does not remove
the expansion so much as make it streamable, so peak memory becomes a chosen chunk size
rather than a function of the table.

## 2. The interface the ICL stage requires

Any replacement has to produce what `ICLearning` already expects, because the point is
to keep its 27.3 M pretrained parameters.

- **Shape** `(B, T, 512)`, where 512 = `embed_dim` (128) x `row_num_cls` (4).
- **Distribution.** `RowInteraction` applies `out_ln = LayerNorm(128)` to each of the 4
  CLS-token outputs and then flattens (`interaction.py`, `_aggregate_embeddings`). So
  each 128-wide chunk of the 512 arrives approximately zero-mean and unit-variance,
  times a learned affine. A replacement must match that, not merely the shape.
- **Targets enter later.** `ICLearning._icl_predictions` adds a y-embedding to the
  context rows of R itself. Note the pretrained column stage is also target-aware
  (`col_target_aware=True`), so under the attention encoder the context rows'
  representations already depend on y before this point. Rung 4 drops that.
- **Row roles.** Test rows attend to context rows only, via `train_size`. An encoder
  that computes context statistics must take them from context rows alone.

## 3. The ladder

"Remove column/row attention" is too coarse to test, because two separable mechanisms
are involved and they are assumed away for different reasons:

- **M1, `ColEmbedding`** -- a context- and target-conditioned embedding of each cell
  value, functionally a learned adaptive per-column quantile transform. It is what makes
  a raw scalar in a never-before-seen column interpretable.
- **M2, `RowInteraction`** -- feature-to-feature interaction within a row, plus
  aggregation of the H cell embeddings into one fixed-length row token.

A fixed-length encoder output assumes away M1 because its dimensions are taken to be
commensurable, and M2 because there is no set of H tokens left to aggregate. Those can
fail independently, so the rungs are worth distinguishing:

| rung | M1 | M2 | what it prices | ceiling at H=62 |
|---|---|---|---|---|
| 0 | full | full | baseline | -- |
| 1 | full | pooled | do within-row feature interactions matter beyond aggregation? | ~50% |
| 2 | `in_linear` only | full | does context-adaptive per-column embedding matter? | ~33% |
| 3 | statistics code | pooled | can closed-form column statistics stand in for M1? | ~83% |
| 4 | none | none | **the fixed-width assumption, stated exactly** | ~83% |

The ceiling column credits the removed stage with zero replacement cost and is computed
from the H=62 row in section 1.

**This branch builds rung 4**, because it is the rung that answers the question. Rung 3
is a hedge against M1 mattering, and it only earns its complexity if rung 4 loses. Rung
2 is the cheap probe that isolates M1 on its own and needs no new module -- gating
`src = self.tf_col(src, ...)` in `_compute_embeddings` yields `in_linear(features)`
directly, since the checkpoint has `col_affine=False`.

## 4. What rung 4 is

`ProjectionEncoder` (`src/tabicl/_model/projection.py`): pad or truncate the row to
`proj_max_features`, an MLP to `embed_dim * row_num_cls`, then `LayerNorm(embed_dim)`
per CLS chunk to meet the section-2 contract. Rows are independent, so inference streams
them in chunks and nothing larger than the output is materialized.

Selected by `TabICL(row_encoder="projection")`, which **builds neither attention stage**,
so the resulting checkpoint carries no `col_embedder` or `row_interactor` weights. The
default stays `"attention"` and that path is unchanged;
`tests/test_prediction_consistency.py` passes against its stored `.npy` references.

`TabICL.with_pretrained_icl(checkpoint, ...)` builds a projection model holding a
checkpoint's ICL weights and a fresh encoder, which no single `load_state_dict`
expresses: the checkpoint has keys this model does not want and lacks the ones it needs.
It refuses overrides that would change an ICL-shaping parameter.

### What it removes, all at once

Four things, and a drop in accuracy cannot be attributed to any one of them without the
intermediate rungs:

1. **Context conditioning** -- a cell's embedding no longer depends on its column's
   distribution across the context rows.
2. **Target awareness** -- y reaches the model only where `ICLearning` adds it.
3. **Feature interaction** -- no attention across a row's features.
4. **Column-order invariance** -- each input position has its own weights.

(4) is a correctness requirement, not just an accuracy cost. `TabICLRegressor` builds
its 8 members from 4 feature shuffles crossed with 2 normalization methods
(`EnsembleGenerator`, `norm_methods=["none","power"]`), and the shuffle is applied by
**physically permuting the columns** before the model sees them
(`preprocessing.py:1182,1200,1215`, `X_preprocessed[:, feat_shuffle]`). Against
per-position weights that permutation is not a symmetry: every member after the first
would feed each weight column a different feature. That corrupts predictions rather than
diversifying them.

So `feat_shuffle_method="none"` is **mandatory** for a projection checkpoint, not a
tuning choice, and `TabICLRegressor.fit` now raises if it is anything else. With it, the
`Shuffler` returns a single identity pattern, so the ensemble is exactly one member per
entry in `norm_methods` -- 2, with `n_estimators` above that having no effect. That is a
further 4x saving *and* a loss of whatever variance reduction the shuffles bought; the
two must be priced separately or one will be read as the other.

### Measured cost

Encoder and full forward at the section-1 shape, with `proj_max_features = H`, against
the attention model's totals from the same table:

| H | encoder params | encoder | full forward | peak GPU | attention forward | attention peak | time | memory |
|---|---|---|---|---|---|---|---|---|
| 23 | 0.550 M | 0.003 s | 0.097 s | 1.13 GB | 0.328 s | 4.81 GB | 3.4x | 4.2x |
| 62 | 0.589 M | 0.003 s | 0.096 s | 1.14 GB | 0.604 s | 11.57 GB | 6.3x | 10.1x |
| 100 | 0.628 M | 0.003 s | 0.098 s | 1.15 GB | 0.878 s | 18.15 GB | 9.0x | 15.7x |
| 200 | 0.731 M | 0.004 s | 0.099 s | 1.18 GB | 1.938 s | 30.09 GB | 19.6x | 25.6x |
| 720 | 1.263 M | 0.005 s | 0.098 s | 1.29 GB | 6.081 s | 45.19 GB | 62.2x | 35.0x |

The encoder costs 3-5 ms at every width, so the full forward is the ICL stage and
almost nothing else -- flat at ~0.097 s and ~1.15 GB from H=23 to H=720. Feature width
has stopped being a cost at all, which is the whole point of the rung: the model's
remaining work does not see it.

The ICL weights load byte-identically (0 of 223 tensors differ), and the encoder's
output satisfies the contract: per-128-chunk mean -0.0000, sd 1.0037.

**These are the ceiling, not a result.** The encoder is untrained. What they say is what
the saving would be *if* training succeeds; the architecture being faster is not
evidence that it works.

## 5. Training

Infrastructure that already exists and does not have to be built: `tabicl.train`
pretrains from the synthetic prior with `--freeze_col/--freeze_row/--freeze_icl`
(`train/_train_config.py`), and `tabicl._finetune` finetunes with the same flags
(`_finetune/base.py:288-290`), covered by `tests/test_finetune_freezing.py`.

This branch adds `--row_encoder`, `--proj_max_features` and `--freeze_proj`, and
`_run.py` refuses `--proj_max_features` below `--max_features` rather than silently
truncating the widest prior tables.

The run that answers the question: **the prior, with the ICL stage frozen**
(`--row_encoder projection --freeze_icl true`), against the real quantile loss. Freezing
ICL is what keeps the result interpretable -- it asks whether a projected representation
is readable by the ICL stage as trained, not whether some jointly trained pair of
networks works.

**The environment cannot train yet.** `ppb-inference`, which ran every benchmark here,
lacks `xgboost` and `transformers`, so `tabicl.prior` will not import and
`tests/test_finetune_freezing.py` will not collect. Training needs the `pretrain` extra
(`transformers`, `xgboost`, `wandb`) in an environment of its own -- not added to
`ppb-inference`, so the measured inference numbers stay reproducible. Until that exists,
the new `--row_encoder` flags are wired but unexercised end to end.

### Feature width becomes a model parameter

This is rung 4's largest hidden cost, and it is worth stating separately from accuracy.

**In attention mode no parameter depends on H.** `ColEmbedding.in_linear` maps a single
cell value (or a group of 3) to `embed_dim`, `y_encoder` is `Linear(1, embed_dim)`, and
`RowInteraction.cls_tokens` is `(num_cls, embed_dim)`. Nothing is sized by feature count,
which is why the pretrained checkpoint reads 23, 62, 720 and 2,048 columns without
modification -- `tabicl_monroe` (720) and `tabicl_chemeleon` (2,048) are both in
`benchmarks/ppb_inference/inference_speed_per_1M.csv`.

`ProjectionEncoder` introduces the first H-sized parameter in the model,
`Linear(proj_max_features, d_model * ff_factor)`. A model built at
`proj_max_features=720` cannot read a 721-column table; wider input raises rather than
being truncated, since dropping columns would yield a plausible wrong number. Giving up
arbitrary width is part of the assumption being priced, not an implementation detail: it
is the difference between a tabular foundation model and a model for one feature layout.

It also changes how the prior should be configured. The prior emits X of shape
`(seq_len, max_features)` zero-padded beyond a per-table `d`, and the attention stages
mask that padding exactly. The projection encoder has no `d` and no mask, so a padding
zero and a real zero are indistinguishable to it. Rather than reintroduce masking --
having no notion of an absent feature is part of the assumption -- train with
`--min_features` equal to `--max_features` equal to `proj_max_features`, so the encoder
sees the deployment layout and spends no capacity on widths it will never meet. That
makes it a per-width model, which is what the paragraph above already concedes.

The prior's `--max_features` is a plain flag with default 100 and no upper bound
(`_dataset.py:131` asserts only `min_features <= max_features`), so training wider is a
matter of compute, not possibility. The real constraint is that the prior must generate
tables at least as wide as the deployment width, or the projection's later input
positions never receive gradient. Note `adjust_max_features` clamps width for long
sequences, but only above `seq_len > 10240`, so it never fires at the default
`--max_seq_len 1024`.

One note on method: distillation onto the pretrained R is *not* the right first move
here, as it was for rung 3. R is a function of the context distribution and a fixed
projection of one row cannot match it, so the distillation error is bounded below by
something that has nothing to do with whether the ablation works.

## 6. Remaining code changes

- **Checkpoint** carrying the new config keys. `_sklearn/regressor.py::_load_model`
  already accepts a local `model_path`, so no loader change is needed.
- **`proppredict`** (`feature/tabicl-inference`): `models/tabicl_model.py::checkpoint_info`
  always calls `hf_hub_download(repo_id=HF_REPO_ID, ...)` with no local-file branch, and
  `fit_fold` checks the resulting sha256 against the recorded run. Evaluating a locally
  trained checkpoint needs `checkpoint_version` to accept an existing path, still hashing
  and recording it. This is the only change the harness needs.

## 7. Evaluation

- **PPB performance.** Re-run the 5x5 folds for `tabicl_monroe_pca62` and compare R2, MAE
  and Spearman against `benchmarks/ppb_performance/metrics_table.csv` through the same
  Westfall-Young maxT bootstrap over test compounds used there.
- **The assumption's boundary.** Run rung 4 on `tabicl_rdkit` as well -- 200 raw
  physicochemical descriptors on heterogeneous scales, 14 of them zero-variance. The PPB
  embedding targets are PCA projections of a learned encoder, which is a **favorable case
  for the assumption, not a neutral one**: orthogonal dimensions with ordered variance,
  on top of TabICL's own normalization. If rung 4 holds on PCA inputs and collapses on
  rdkit, that is the boundary measured rather than argued, and it is the difference
  between "fixed-width grounded embeddings do not need M1" and "nothing needs M1".
- **Inference speed.** Re-run `benchmarks/ppb_inference` for a new h/1M row, reporting
  the 2-member and 8-member ensembles separately.
- **Regression.** `tests/test_prediction_consistency.py`, `tests/test_inference.py`,
  `tests/test_inference_with_cache.py` and `tests/test_oom_retry.py` with the flag off
  (36 passed, 25 skipped at the time of writing), plus
  `tests/test_projection_encoder.py`.

**Both optimizers are safe with a frozen ICL stage.** `configure_optimizer` hands every
parameter to AdamW or Muon regardless of `requires_grad`, but both skip parameters whose
grad is None (`_muon.py:124,179`), so frozen weights are neither decayed nor updated.
Checked because silent corruption of the pretrained stage would invalidate the whole
experiment.

**Prerequisite, and it is not optional.** TabICL derives inference batch sizes from GPU
memory free at predict time, and varying batch size alone moves individual predictions by
up to 1.4e-02 on a target of sd ~0.6 (measured under fixed code, safety_factor 0.9 vs
0.05). Any before/after comparison must pin `inference_config` on both sides, or that
drift will be read as an effect of the architecture change.

## 8. Out of scope

The ICL stage's own architecture; the synthetic prior; the preprocessing and
normalization pipeline; the classification, finetuning and forecasting paths; and
anything that changes the numbers already reported in
`benchmarks/ppb_performance/report.md`.
