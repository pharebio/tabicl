# PPB: TabICL vs chemprop vs Random Forest

Date: 2026-11-02. Extended 2026-11-04 (`tabicl_morgan`), 2026-11-05 (CheMeleon PCA variants, Monroe), 2026-11-06 (Random Forest on Monroe embeddings). Report text updated 2026-11-09 to cover all 15 models, and the library inference of `tabicl_monroe_pca100` and the three `rf_monroe` variants measured the same day. Extended later on 2026-11-09 with `chemprop_monroe` and `chemprop_monroe_pca200` — chemprop taking the Monroe embedding as molecule-level descriptors — bringing the performance benchmark to 17 models. Their library inference was measured 2026-11-11, so both benchmarks covered all 17. Extended 2026-11-16 with `tabicl_monroe_pca62` and `tabicl_monroe_pca23`, whose widths come from the dimensionality analysis in `representation_dimensionality.csv`, bringing the performance benchmark to 19 models. Their library inference was measured 2026-11-18, so both benchmarks again cover every model.

## What was run

Base model: `ppb_batches_1to5_testing_log10_unbound` (PPB batches 1–5, single task, no phb640/WuXi augmentation). Target: `ppb_mean_plasma_log10_frac_unbound_value`. Split: 5x5 CV (25 splits) with one fixed Butina-cluster held-out test set.

TabICL v2.2.0 (`TabICLRegressor`, checkpoint `tabicl-regressor-v2-20260212.ckpt`, sha256 `0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a`), installed from `pharebio/tabicl@0dbff3ec` (= upstream tag v2.2.0). Default estimator settings (`n_estimators=8`, `random_state=42`). No tuning.

TabICL has no training step. For each of the 25 splits, the `train` partition (~5,150 rows) is the context and the test set is predicted. `val` is not used. In these splits train ∪ val is the same 6,298 rows in every split, so using train+val as context would give 25 identical folds.

### Encoders

- **CheMeleon**: frozen foundation encoder, mean aggregation, 2,048-d.
- **Monroe**: GRIT graph transformer pretrained on PM6 (~81M molecules) and PCBA (1.56M), 720-d graph embedding, 58.6M parameters. Repo `blazejba/monroe@de075eab`, bundled checkpoint `checkpoint/weights.pt` sha256 `df996e8e98b12a3cda5fe5acbef3e1e9a4ea31946d3736461cd07e53a5bf205a` (hidden_dim 720, 10 layers, 10 heads, walk_len 16, stereo edges on). Run from its own venv at `/data/andrew/envs/monroe` via `all_batches/scripts/embed_monroe.py`; the `ppb-inference` env and its pinned RDKit were left untouched. Embedding the 7,032 PPB molecules took 2.2 min (53.9 rows/s) with 0 failures; 5 molecules could not get an MMFF/UFF 3D conformer and fell back to a 2D conformation.

### PCA variants

PCA is fit on the 6,298 non-test rows only, then applied to all rows, by the existing `build_chemeleon_pca_features.py --source/--prefix`. Explained variance: `chemeleon_pca720` 0.9762, `monroe_pca200` 0.9946, `monroe_pca100` 0.9717, `monroe_pca62` 0.9337, `monroe_pca23` 0.7611. (`chemeleon_pca200`/`pca100` are the 2026-11-04 variants; `monroe_pca62`/`pca23` the 2026-11-16 ones.) The two 2026-11-16 variants take their widths from `representation_dimensionality.csv`: 62 is Monroe's parallel-analysis count and 23 the participation ratio of its covariance spectrum. Their components match `monroe_pca100`'s leading ones to |cos| ≥ 0.996 with an identical mean vector, so all five Monroe variants are nested truncations of one projection rather than independent fits.

### chemprop with Monroe descriptors

chemprop v2 takes a molecule-level feature matrix through `--descriptors-path` and concatenates it (`X_d`) to the learned graph representation before the FFN head, scaling it by default with a scaler stored in the checkpoint. So `chemprop_monroe` is the existing chemprop baseline — same CheMeleon foundation, same 25 splits, same hyperparameters, recovered from the baseline's own ClearML task rather than reconstructed — with the 720-d Monroe embedding added as descriptors. It is not a replacement of chemprop's encoder: CheMeleon can serve as chemprop's encoder because it is a chemprop-compatible D-MPNN checkpoint, whereas Monroe is a GRIT graph transformer and cannot be loaded as one.

chemprop trains on `ppb_batches_1to5_testing_log10_unbound_parsed.csv` (7,039 rows), not on the TabICL feature set's `aligned_parsed.csv` (7,032 rows): the 7 rows that fail RDKit/Morgan featurization are absent from the latter, and `--descriptors-path` requires one row per CSV row. The descriptors were therefore built for the 7,039-row CSV, keeping chemprop's own training set and its 735-compound test set unchanged.

Re-embedding all 7,039 rows would **not** have reproduced `monroe.npz`: conformer generation is stochastic, and a second pass over the same molecules differed by mean 0.009 and at most 0.53 per component, against an embedding sd of 0.35 (minimum row correlation 0.990). That would have confounded `chemprop_monroe` vs `tabicl_monroe` with the featurizer's randomness. `embed_monroe.py` gained `--reuse_npz/--reuse_csv`, which copies the rows an existing feature file already covers and embeds only the remainder; the 7,032 shared rows in `monroe_parsed7039.npz` are bit-identical to `monroe.npz` and only the 7 extra molecules were embedded. All 7 succeeded, 5 of them on the 2D-conformer fallback. For the same reason `build_chemeleon_pca_features.py` gained `--project_with`, which reuses the stored PCA mean and components instead of refitting on the larger row set, so `chemprop_monroe_pca200` sees the same 200 components as `tabicl_monroe_pca200` (shared rows agree to 5e-06, float32 rounding).

### Models

| Model | Features | Columns |
|---|---|---|
| `tabicl_rdkit` | RDKit-v1 descriptors | 200 |
| `tabicl_rdkit_morgan` | RDKit-v1 + Morgan r=2 2048-bit (same features as the RF model) | 2,248 |
| `tabicl_morgan` | Morgan r=2 2048-bit | 2,048 |
| `tabicl_chemeleon` | Frozen CheMeleon encoder | 2,048 |
| `tabicl_chemeleon_pca720` | CheMeleon → PCA | 720 |
| `tabicl_chemeleon_pca200` | CheMeleon → PCA | 200 |
| `tabicl_chemeleon_pca100` | CheMeleon → PCA | 100 |
| `tabicl_monroe` | Monroe encoder | 720 |
| `tabicl_monroe_pca200` | Monroe → PCA | 200 |
| `tabicl_monroe_pca100` | Monroe → PCA | 100 |
| `tabicl_monroe_pca62` | Monroe → PCA, k from parallel analysis | 62 |
| `tabicl_monroe_pca23` | Monroe → PCA, k from the participation ratio | 23 |
| `chemprop_monroe` | chemprop (CheMeleon) + Monroe descriptors | 720 |
| `chemprop_monroe_pca200` | chemprop (CheMeleon) + Monroe → PCA descriptors | 200 |
| `rf_monroe` | Monroe encoder, Random Forest | 720 |
| `rf_monroe_pca200` | Monroe → PCA, Random Forest | 200 |
| `rf_monroe_pca100` | Monroe → PCA, Random Forest | 100 |

Comparison models already trained on the same 25 splits: chemprop (CheMeleon, fine-tuned per fold) and `random_forest` (RDKit-v1 + Morgan, 100 trees).

The three `rf_monroe*` runs come from `all_batches/scripts/train_ppb_rf_cv.py`, written for this comparison: it runs the same 25-fold loop over the same `splits_aligned.json`, with `n_estimators=100, random_state=42` and train+val both handed to the forest with a `split` column, matching the existing `random_forest` baseline rather than TabICL's train-only context.

All 19 models are evaluated on the 734 test compounds common to all of them. The chemprop test set has 735 compounds; one (`Nc1cc(Cl)c2n[se]nc2c1[N+](=O)[O-]`) fails RDKit/Morgan featurization.

## Results

Ensemble mean (mean of the 25 fold predictions) vs. measured, n = 734, sorted by MAE:

| Model | MAE | RMSE | R² | Pearson r | Spearman r |
|---|---:|---:|---:|---:|---:|
| tabicl_monroe | 0.353 | 0.535 | 0.637 | 0.799 | 0.840 |
| tabicl_monroe_pca100 | 0.358 | 0.540 | 0.631 | 0.795 | 0.839 |
| tabicl_monroe_pca200 | 0.360 | 0.538 | 0.634 | 0.798 | 0.837 |
| tabicl_monroe_pca62 | 0.362 | 0.545 | 0.623 | 0.790 | 0.834 |
| chemprop_monroe | 0.365 | 0.542 | 0.628 | 0.793 | 0.830 |
| tabicl_monroe_pca23 | 0.370 | 0.554 | 0.611 | 0.783 | 0.822 |
| tabicl_chemeleon_pca720 | 0.385 | 0.572 | 0.586 | 0.766 | 0.790 |
| tabicl_chemeleon_pca200 | 0.393 | 0.588 | 0.561 | 0.750 | 0.781 |
| tabicl_chemeleon_pca100 | 0.394 | 0.590 | 0.560 | 0.749 | 0.777 |
| chemprop | 0.397 | 0.590 | 0.558 | 0.751 | 0.788 |
| tabicl_rdkit | 0.397 | 0.592 | 0.556 | 0.746 | 0.786 |
| chemprop_monroe_pca200 | 0.406 | 0.588 | 0.562 | 0.754 | 0.791 |
| rf_monroe | 0.433 | 0.597 | 0.548 | 0.745 | 0.770 |
| tabicl_chemeleon | 0.437 | 0.635 | 0.489 | 0.700 | 0.732 |
| random_forest | 0.450 | 0.629 | 0.499 | 0.709 | 0.743 |
| tabicl_rdkit_morgan | 0.463 | 0.639 | 0.483 | 0.698 | 0.744 |
| rf_monroe_pca100 | 0.492 | 0.660 | 0.448 | 0.677 | 0.704 |
| rf_monroe_pca200 | 0.510 | 0.675 | 0.423 | 0.659 | 0.676 |
| tabicl_morgan | 0.573 | 0.772 | 0.246 | 0.497 | 0.545 |

Per-fold (25 folds, mean ± std):

| Model | R² | MAE | Pearson r | Spearman r |
|---|---|---|---|---|
| tabicl_monroe | 0.627 ± 0.007 | 0.357 ± 0.004 | 0.793 ± 0.005 | 0.834 ± 0.004 |
| tabicl_monroe_pca100 | 0.616 ± 0.012 | 0.364 ± 0.007 | 0.786 ± 0.007 | 0.829 ± 0.008 |
| tabicl_monroe_pca200 | 0.617 ± 0.010 | 0.367 ± 0.006 | 0.788 ± 0.006 | 0.826 ± 0.007 |
| tabicl_monroe_pca62 | 0.610 ± 0.009 | 0.368 ± 0.005 | 0.783 ± 0.006 | 0.826 ± 0.006 |
| chemprop_monroe | 0.608 ± 0.008 | 0.379 ± 0.007 | 0.782 ± 0.004 | 0.814 ± 0.006 |
| tabicl_monroe_pca23 | 0.599 ± 0.008 | 0.375 ± 0.005 | 0.777 ± 0.005 | 0.815 ± 0.006 |
| tabicl_chemeleon_pca720 | 0.560 ± 0.020 | 0.395 ± 0.008 | 0.750 ± 0.012 | 0.775 ± 0.011 |
| tabicl_chemeleon_pca200 | 0.543 ± 0.014 | 0.400 ± 0.007 | 0.739 ± 0.009 | 0.770 ± 0.009 |
| tabicl_chemeleon_pca100 | 0.543 ± 0.015 | 0.400 ± 0.007 | 0.739 ± 0.009 | 0.768 ± 0.009 |
| tabicl_rdkit | 0.543 ± 0.014 | 0.404 ± 0.006 | 0.738 ± 0.009 | 0.778 ± 0.006 |
| chemprop | 0.539 ± 0.012 | 0.407 ± 0.009 | 0.741 ± 0.006 | 0.776 ± 0.007 |
| chemprop_monroe_pca200 | 0.543 ± 0.011 | 0.416 ± 0.007 | 0.745 ± 0.007 | 0.781 ± 0.009 |
| rf_monroe | 0.535 ± 0.008 | 0.438 ± 0.005 | 0.737 ± 0.005 | 0.761 ± 0.005 |
| tabicl_chemeleon | 0.477 ± 0.010 | 0.442 ± 0.004 | 0.692 ± 0.007 | 0.724 ± 0.008 |
| random_forest | 0.481 ± 0.012 | 0.458 ± 0.007 | 0.695 ± 0.008 | 0.730 ± 0.010 |
| tabicl_rdkit_morgan | 0.466 ± 0.008 | 0.469 ± 0.005 | 0.686 ± 0.006 | 0.734 ± 0.010 |
| rf_monroe_pca100 | 0.428 ± 0.012 | 0.499 ± 0.006 | 0.659 ± 0.010 | 0.684 ± 0.009 |
| rf_monroe_pca200 | 0.402 ± 0.011 | 0.517 ± 0.005 | 0.640 ± 0.009 | 0.655 ± 0.008 |
| tabicl_morgan | 0.239 ± 0.010 | 0.575 ± 0.004 | 0.491 ± 0.010 | 0.537 ± 0.011 |

### Tukey HSD over the 25 folds

Two-way ANOVA blocked on fold, FWER 0.05, all 171 pairs in `tukey_intervals_pairs.csv`. Differences below are model A minus model B on the per-fold means; `p < 1e-13` is the floor the adjusted p-values are clipped to.

**Read these p-values with the design in mind: they do not cover test-set sampling.** All 25 CV folds are scored on the *same* 734 compounds — only the train/val split and training stochasticity change — so the ANOVA residual is training-replicate variance alone. The computation is correct (residual MSE and df reproduce statsmodels' two-way ANOVA exactly, and the comparison intervals satisfy non-overlap ⟺ HSD rejection), but it answers "do these models differ across training replicates **on this test set**", not "would this difference hold on other compounds". The second question is the one a reader usually has, and its uncertainty is **11–21× larger**: the Tukey comparison interval for R² is ±0.006, while a bootstrap over the 734 test compounds gives each model a 95% interval about 0.12 wide.

So the benchmark now reports both, and `tukey_intervals_pairs.csv` carries `p_boot_maxT` beside `p_tukey` for every pair. The companion is a **paired** bootstrap — both models scored on the same resampled compounds, which is what keeps a difference far better determined than either metric — adjusted over the same family of 171 pairs by Westfall–Young maxT, so the two columns are comparable. (The unadjusted bootstrap would not be: Tukey is family-adjusted, and comparing it against raw per-pair p-values understates the disagreement by half.)

**The two figures are the same chart under the two definitions of "tied with the best model".** `tabicl_monroe` is best on all three metrics. Under Tukey, 0 / 1 / 1 other models (MAE / R² / Spearman) are tied with it: `tabicl_monroe_pca200` on R², `tabicl_monroe_pca100` on Spearman. Under the bootstrap, 5 / 5 / 4 are — every TabICL Monroe PCA variant down to `pca23`, plus `chemprop_monroe`, on MAE and R²; the same set without `pca23` on Spearman. Nothing moves in the ranking — only how much of it the data can resolve.

**Effect size separates the two tests, but not sharply enough for a rule of thumb.** Of 513 comparisons (171 pairs × 3 metrics), 458 are Tukey-significant and 333 survive the adjusted bootstrap; the 125 that do not (24%) have a median |difference| of 0.025, against 0.089 for those that do. The two distributions overlap: 55 Tukey-significant comparisons with |difference| above 0.03 fail the bootstrap, and the largest failing difference is 0.079. Each claim below therefore carries its own p-values rather than relying on an effect-size threshold. Nothing reverses direction — the ranking is unchanged.

The family grew from 136 to 171 pairs when `tabicl_monroe_pca62` and `tabicl_monroe_pca23` were added, so both columns are adjusted over a larger family than in earlier revisions of this report and every p-value here has been restated from the regenerated tables.

**Monroe beats CheMeleon as an encoder, at every width tested.** `tabicl_monroe` vs the best CheMeleon variant `tabicl_chemeleon_pca720`: R² +0.068, MAE −0.038, Spearman +0.060, all p < 1e-13 (maxT 0.032, 0.001, < 0.001). At matched width the gap is the same size: `monroe_pca200` vs `chemeleon_pca200` R² +0.073, MAE −0.033; `monroe_pca100` vs `chemeleon_pca100` R² +0.073, MAE −0.036; all p < 1e-13, and all six survive the bootstrap (maxT ≤ 0.035).

**Six models beat chemprop on every metric, and all six use Monroe.** The five TabICL Monroe models — `tabicl_monroe` vs chemprop R² +0.089, MAE −0.050, Spearman +0.058; `monroe_pca200` +0.078, −0.040, +0.050; `monroe_pca100` +0.077, −0.043, +0.053; `monroe_pca62` +0.071, −0.039, +0.049; `monroe_pca23` +0.061, −0.032, +0.039 — and `chemprop_monroe` itself (below); all p < 1e-13. Under the bootstrap the R² and MAE gaps hold for every variant except `pca23` (maxT 0.26 and 0.10), and the Spearman gap holds for all six. No CheMeleon variant beats chemprop on all three: the best, `chemeleon_pca720`, is ahead on R² (+0.021, p = 4.7e-9) and MAE (−0.012, p = 8.3e-11), neither surviving the bootstrap (maxT 1.00 and 0.98), and level on Spearman (−0.002, p = 1.0).

**The embedding helps a trained network too, which rules out an in-context-specific explanation.** Note what `chemprop_monroe` is: every chemprop model here is fine-tuned from CheMeleon (`from-foundation = CHEMELEON`, encoder output 2,048-d, so the FFN input is 2,048 + 720 = 2,768), which makes it the only model in the benchmark using *both* foundation models, where `rf_monroe` and the `tabicl_monroe` variants use Monroe alone. The finding is therefore that Monroe adds to CheMeleon, not that it replaces it. `chemprop_monroe` vs `chemprop` — same architecture, same folds, same hyperparameters, the only change being the 720-d Monroe embedding as descriptors — is R² +0.069, MAE −0.028, Spearman +0.037, all p < 1e-13. Against `rf_monroe`, which sees the identical embedding, it is R² +0.072, MAE −0.058, Spearman +0.052 (p < 1e-13), so the embedding is worth far more to a trained network than to a forest: the same features move a Random Forest from R² 0.481 to 0.535 and chemprop from 0.539 to 0.608.

**`chemprop_monroe` matches the compressed TabICL Monroe models on R²; the MAE and ranking gaps are Tukey-only.** *(Bootstrap: none of the differences below survives test-compound resampling — maxT p ≥ 0.53 — so on this evidence the models are not separable on any metric.)* vs `tabicl_monroe_pca100`: R² −0.008 (p = 0.48, not significant), MAE +0.015 (p < 1e-13), Spearman −0.016 (p = 6.9e-12). vs `tabicl_monroe_pca200`: R² −0.009 (p = 0.30, not significant), MAE +0.012 (p = 4.2e-11), Spearman −0.013 (p = 1.2e-7). vs `tabicl_monroe_pca62`: R² −0.002 (p = 1.0), MAE +0.011 (p = 1.1e-9), Spearman −0.012 (p = 7.7e-7). vs `tabicl_monroe_pca23` it is indistinguishable on all three (R² +0.008, p = 0.40; MAE +0.004, p = 0.42; Spearman −0.002, p = 1.0). Against full-width `tabicl_monroe` it loses on all three under Tukey (R² −0.019, p = 1.0e-7; MAE +0.022; Spearman −0.021), none of which survives the bootstrap (maxT 0.98, 0.052, 0.088). So the in-context model keeps an advantage on this data, but on R² alone it is now a tie with a trained network given the same chemistry.

**PCA to 200 components is nearly free for TabICL and gives back almost the whole gain for chemprop.** `chemprop_monroe_pca200` vs `chemprop_monroe`: R² −0.064, MAE +0.037, Spearman −0.033, all p < 1e-13 — which leaves it indistinguishable from plain `chemprop` on R² (+0.005, p = 0.99) and Spearman (+0.004, p = 0.86), and nominally *worse* on MAE (+0.009, p = 6.2e-6 — but maxT p = 0.68, so this one does not survive resampling). The same projection cost `tabicl_monroe` 0.011 R² (p = 0.063, not significant). The R² collapse itself is robust (−0.064, maxT p < 0.001, as are its MAE and Spearman counterparts); it is only the MAE sign that is Tukey-only. Three models now consume the identical 200 components and rank in a wide fixed order — `tabicl_monroe_pca200` R² 0.617, `chemprop_monroe_pca200` 0.543, `rf_monroe_pca200` 0.402 — so how much of the embedding survives compression is a property of the consumer, not of the projection.

**PCA no longer helps once the encoder is already narrow.** Reducing CheMeleon from 2,048 to 720 was a large gain (R² +0.083, MAE −0.046, p < 1e-13). Reducing Monroe from 720 to 200 or 100 is a small loss, and *none of it survives test-compound resampling*: `monroe` → `monroe_pca200` MAE +0.010 (p = 3.2e-7, maxT 0.77), R² −0.011 (p = 0.063), Spearman −0.008 (p = 0.022, maxT 0.88); `monroe` → `monroe_pca100` MAE +0.007 (p = 0.0023, maxT 0.98), R² −0.011 (p = 0.027, maxT 0.98). On 734 compounds, compressing Monroe for TabICL is free within the resolution of the test set. `monroe_pca200` and `monroe_pca100` are indistinguishable from each other on all three metrics (p ≥ 0.96). This holds for TabICL only — see the chemprop result above.

**The widths predicted from the spectrum bracket where compression starts to cost something.** `tabicl_monroe_pca62` takes k from Monroe's parallel-analysis count — 62 directions above sampling noise, 93.4% of the covariance variance — and `tabicl_monroe_pca23` from the participation ratio of its covariance spectrum, 22.6 rounded to 23 components, 76.1% of the variance. At k = 62 the model is indistinguishable from `monroe_pca100` and `monroe_pca200` on all three metrics (p ≥ 0.73), and its deficit against full-width `tabicl_monroe` — R² −0.017, MAE +0.011, Spearman −0.008 — is Tukey-significant (p = 6.9e-6, 1.7e-8, 0.0065) but does not survive the bootstrap (maxT 0.85, 0.66, 0.64). At k = 23 the deficit roughly doubles — R² −0.028, MAE +0.018, Spearman −0.019, all p < 1e-13 — and the ranking loss now survives test-compound resampling (Spearman maxT 0.016; MAE 0.073, R² 0.29). Parallel analysis marked the floor correctly: 62 components cost nothing this test set can resolve, 23 cost measurable ranking accuracy.

**23 Monroe components still beat every non-Monroe model.** `tabicl_monroe_pca23` vs chemprop R² +0.061, vs `tabicl_rdkit` +0.057, vs `rf_monroe` +0.064, all p < 1e-13, and it is indistinguishable from `chemprop_monroe` on all three metrics (R² +0.008 p = 0.40, MAE +0.004 p = 0.42, Spearman −0.002 p = 1.0). A 23-column feature set in context matches a trained network reading 2,768 inputs.

**A Monroe embedding is better input than RDKit+Morgan even for a Random Forest.** `rf_monroe` vs `random_forest`: R² +0.055, MAE −0.020, Spearman +0.032, all p < 1e-13, with 720 features against 2,248. *None of the three survives test-compound resampling* (maxT 0.30, 0.83, 0.46), so the forest-level gain is the weakest of the Monroe results: it is consistent across training replicates and not resolvable across compounds.

**The in-context model contributes about four times what the encoder swap does.** On the same folds, swapping RDKit+Morgan for Monroe under a Random Forest is worth 0.020 MAE; swapping the Random Forest for TabICL on the same Monroe features is worth 0.080 MAE (`tabicl_monroe` vs `rf_monroe`: R² +0.092, Spearman +0.073, all p < 1e-13). That ratio is specific to the Random Forest being the alternative: swapping it for chemprop instead recovers 0.058 of that 0.080, so most of the gap is the forest, not the absence of in-context learning.

**`rf_monroe` reaches plain chemprop's R² but not its MAE or ranking, under Tukey only.** R² −0.003 (p = 1.0, not significant), MAE +0.030 (p < 1e-13), Spearman −0.015 (p = 3.0e-10); the MAE and Spearman gaps do not survive the bootstrap (maxT 0.43 and 1.00). Against `tabicl_rdkit`: R² −0.007 (p = 0.65, not significant), MAE +0.034 (maxT 0.052), Spearman −0.017 (maxT 0.99). It does not reach `chemprop_monroe`, which is 0.072 R² above it on the same embedding.

**PCA is harmful for the Random Forest, in the opposite direction to TabICL.** `rf_monroe` → `rf_monroe_pca200` costs R² 0.133 and MAE 0.080; → `rf_monroe_pca100` costs R² 0.107 and MAE 0.062; both p < 1e-13 and both surviving the bootstrap; `rf_monroe_pca200` falls below the `random_forest` baseline (R² −0.079, p < 1e-13, but maxT 0.16). The same projection cost TabICL at most 0.010 MAE. A forest splits on individual axes, and the PCA rotation spreads the informative directions across many of them.

**Findings from the original six models hold, but their adjusted p-values are not the ones earlier versions of this report quoted** — Tukey's adjustment depends on the number of groups, and there are now 19 rather than 6, 15 or 17, so every p-value has moved again. Restated from the current tables: chemprop and `tabicl_rdkit` remain indistinguishable (R² p = 1.0, MAE p = 0.71, Spearman p = 1.0); `tabicl_rdkit_morgan` is still worse than `random_forest` on the same inputs (R² −0.015, p = 2.3e-4; MAE +0.011, p = 1.8e-9), though neither gap survives the bootstrap (maxT 1.00 and 0.99); `tabicl_chemeleon` vs `random_forest` is R² indistinguishable (p = 1.0) with lower MAE for TabICL (−0.016, p < 1e-13, maxT 0.98); `tabicl_morgan` is below every other model on every metric (all 54 of its comparisons at p < 1e-13, and all at maxT ≤ 0.001). Adding Morgan bits to RDKit descriptors lowers TabICL R² by 0.077 and raises MAE by 0.065 (p < 1e-13); the six-model report gave 0.052 for that R² gap, which does not match the per-fold means it was drawn from (0.543 − 0.466 = 0.077).

### Runtime

Wall-clock for the full 25-fold run, one RTX 6000 Ada (RF on 32 CPU cores), including ClearML setup and teardown:

| Model | 25 folds |
|---|---:|
| `tabicl_monroe` (720) | 4.4 min |
| `tabicl_monroe_pca200` | 1.7 min |
| `tabicl_monroe_pca100` | 1.0 min |
| `tabicl_chemeleon_pca720` | 4.3 min |
| `tabicl_chemeleon_pca200` | 1.8 min |
| `tabicl_chemeleon_pca100` | 1.4 min |
| `rf_monroe` | 3.5 min |
| `rf_monroe_pca200` | 0.9 min |
| `rf_monroe_pca100` | 0.5 min |
| `chemprop_monroe` | 49.2 min |
| `chemprop_monroe_pca200` | 54.3 min |

The two chemprop runs are not comparable with the rows above them: they shared GPU 1 with an unrelated 25-fold chemprop CV on 256k rows that was running throughout, so their wall-clock includes contention. They are also a different kind of work — 25 fine-tuning runs of up to 50 epochs each, against TabICL's 25 forward passes.

Per fold for the original four TabICL sets: about 4 s (rdkit), 44 s (rdkit+morgan), 28 s (morgan), 33 s (chemeleon). Peak GPU memory for the 2,000+ column sets was about 25 GB. No offloading. Monroe embedding of the 7,032 PPB molecules is a one-off 2.2 min, not counted above.

## Inference cost

CV runtime is not the cost that matters for library scoring; `benchmarks/ppb_inference/` measures that separately on yu14M shards, and covers all 19 models — `tabicl_monroe_pca62` and `tabicl_monroe_pca23` were measured 2026-11-18 on the same two shards, same hardware and same chunk as the rest (figure: `inference_speed_per_1M.png`, whose lower panel expands the fourteen models under 4 h, since on a single linear axis they are slivers next to the 100-hour feature sets). Hours to score 1M compounds with the 25-model ensemble, every model at predict chunk 50,000:

| Model | Featurization | Model | Total | At its own best chunk |
|---|---:|---:|---:|---:|
| `random_forest` | 0.42 | 0.03 | 0.45 | – |
| `chemprop` | 0.04 | 1.01 | 1.05 | – |
| `rf_monroe_pca200` | 1.26 | 0.01 | 1.26 | – |
| `rf_monroe_pca100` | 1.26 | 0.01 | 1.26 | – |
| `rf_monroe` | 1.26 | 0.01 | 1.27 | – |
| `tabicl_chemeleon_pca100` | 0.13 | 1.16 | 1.29 | 1.29 (50,000) |
| `tabicl_monroe_pca23` | 1.26 | 0.39 | 1.65 | 1.65 (50,000) |
| `tabicl_monroe_pca62` | 1.26 | 0.78 | 2.04 | 2.04 (50,000) |
| `tabicl_rdkit` | 0.23 | 1.87 | 2.10 | 2.10 (50,000) |
| `chemprop_monroe_pca200` | 1.29 | 0.98 | 2.28 | – |
| `chemprop_monroe` | 1.30 | 0.98 | 2.28 | – |
| `tabicl_monroe_pca100` | 1.26 | 1.15 | 2.41 | 2.41 (50,000) |
| `tabicl_chemeleon_pca200` | 0.13 | 2.74 | 2.87 | 2.62 (25,000) |
| `tabicl_monroe_pca200` | 1.26 | 2.67 | 3.93 | 3.66 (25,000) |
| `tabicl_chemeleon_pca720` | 0.13 | 10.90 | 11.03 | 11.03 (50,000) |
| `tabicl_monroe` | 1.26 | 11.06 | 12.32 | 12.32 (50,000) |
| `tabicl_chemeleon` | 0.12 | 102.40 | 102.52 | 39.64 (10,000) |
| `tabicl_morgan` | 0.20 | 107.10 | 107.30 | 44.37 (10,000) |
| `tabicl_rdkit_morgan` | 0.42 | 114.60 | 115.02 | 50.47 (10,000) |

The table and the figure hold the predict chunk fixed at 50,000 for every model, because that is the controlled comparison. The last column gives each model's best over the 2,500–50,000 sweep, which is what it would cost if tuned — but those per-model bests are not comparable with each other: for every narrow feature set the best found was 50,000 itself, the edge of the sweep with the curve still falling, while for `chemeleon` it was a true interior minimum at 10,000. Ranking on a mix of censored and uncensored optima would flatter whichever models happen to have an interior one.

**Monroe's encoder is a floor of 1.26 h/1M under every model that uses it** — 9.5x CheMeleon's 0.13 h, because conformer generation is per-molecule CPU work. It dominates the three forests completely: their own predict stage is 0.01 h, so `rf_monroe` costs 1.27 h to reach an R² indistinguishable from chemprop's at 1.05 h, with worse MAE and Spearman. Paying for the encoder only pays off under the in-context model.

**Among the TabICL models, `tabicl_monroe_pca62` at 2.04 h is the cheapest that beats chemprop on every metric.** Its margins over chemprop are R² +0.071, MAE −0.039 and Spearman +0.049, and all three survive test-compound resampling. `tabicl_monroe_pca23` is cheaper still at 1.65 h — the cheapest TabICL model in the benchmark — and leads chemprop on all three under Tukey, but only its ranking gain survives the bootstrap, so the last 0.39 h is bought by trading a claim that holds on three metrics for one that holds on one. Above `pca62`, `tabicl_monroe_pca100` at 2.41 h is indistinguishable from `tabicl_monroe_pca200` on all three metrics (p ≥ 0.95) at two-thirds the cost, so it, not `pca200`, is the better buy of those two, and full-width `tabicl_monroe` buys 0.011 R² over it for 5x the cost.

**TabICL's model stage is linear in feature width up to 100 columns.** Across the five TabICL Monroe variants it is 0.39 h (k = 23), 0.78 (62), 1.15 (100), 2.67 (200) and 11.06 (720) per 1M. A straight line through the three narrow points — 0.0098 h per column plus a 0.17 h fixed cost — reproduces them to 0.07%, so in that range each additional PCA component costs about 35 s per 1M compounds and nothing else varies. The line breaks above it: k = 200 runs 1.25x it (its 50,000-chunk figure is extrapolated from a 25,000-chunk run) and full-width 720 runs 1.53x. Peak GPU memory follows the same shape — 18.7 GB at k = 23, 30.4 at 62, 34.8 at 100 — making `pca23` the lightest TabICL row in the benchmark. A power law fitted on the k ≥ 100 runs before these two were measured underpredicted them by 13% at k = 62 and 75% at k = 23, which is the fixed cost it had no leverage to see.

**The Monroe descriptor block is free at inference.** `chemprop_monroe` costs 2.28 h/1M and `chemprop_monroe_pca200` 2.28 h, measured 2026-11-11 on the same two yu14M shards (199,978 rows, 25 folds, one RTX 6000 Ada) as every other row. Their predict stages are 0.98 h each — indistinguishable from each other and, within run-to-run noise, from plain chemprop's 1.01 h, although each forward pass concatenates 720 (or 200) extra columns before the FFN. Peak GPU memory is 3.4 GB, the same as plain chemprop. Essentially the whole difference from chemprop's 1.05 h is the Monroe encoder's 1.26 h, charged from the same `monroe_s` in the same shard `timing.json` that `rf_monroe` and `tabicl_monroe` are charged from. That the 720-d and 200-d variants cost the same is the expected result and a check on the measurement: the descriptor width only widens the FFN's input layer.

Across the 17 models measured before 2026-11-18 this made `chemprop_monroe` the cheapest that beats chemprop on all three metrics — 2.28 h against `tabicl_monroe_pca100`'s 2.41 h, with R² statistically tied between them (−0.008, p = 0.49) and TabICL's nominal edge on MAE and ranking Tukey-only, not surviving test-compound resampling (maxT p = 1.00 and 0.99). **`tabicl_monroe_pca62` takes that place at 2.04 h**, and not as a trade: against `chemprop_monroe` it is level on R² (+0.002, p = 1.0) and ahead on MAE (−0.011) and Spearman (+0.012) under Tukey, with no difference between them surviving the bootstrap (maxT ≥ 0.53). So `chemprop_monroe` is now undercut on cost at indistinguishable measured accuracy, and `chemprop_monroe_pca200` costs the same 2.28 h while giving the accuracy back, so it is dominated by both.

Reaching these numbers needed a change to the inference path, since chemprop could not take descriptors through it at all: `run_inference_python_api.prepare_dataloader` now takes an optional `x_d` and builds `MoleculeDatapoint.from_smi(smi, x_d=row)` (every stage downstream already carried `X_d`), and `run_library_inference.py` gained `--descriptor_features`. The path is verified against the CV models rather than assumed: reloading `chemprop_monroe/model_0/best.pt` and predicting its 735 test compounds through it reproduces the chemprop CLI's `test_predictions.csv` to a maximum absolute difference of 5.2e-07, shuffling the descriptor rows moves the predictions by 0.64 on average, and omitting them raises a shape error (the FFN takes 2,768 = 2,048 graph + 720 descriptors) rather than degrading quietly.

**A note on how these two timings were obtained, because the first attempt was wrong.** A CV trainer runs each fold as a fresh `chemprop` subprocess, so between folds its GPU shows no compute process for a few seconds. A wait condition that only asks whether anything currently holds GPU memory can be satisfied inside such a gap, and the 2026-11-10 attempt was: both runs started in inter-fold gaps on a device another job owned and finished 29% and 77% contended, reporting 0.985 and 1.602 h/1M. The 1.602 was visibly impossible — `pca200` has a narrower descriptor block than `monroe` and cannot be 63% slower — which is what gave the contamination away. The runs published here come from `run_chemprop_monroe.sh`, which treats a device as busy whenever a CV trainer claiming it is alive anywhere on the machine (read from its own `--gpus` / `--device` flags), requires five consecutive clear minutes, and samples the device every 20 s during the run; both published runs recorded **0 intruder samples**, and their wall times are 12.4 min each rather than the contended 12.5 and 19.8. The rejected runs are kept beside them as `chemprop_monroe_contended_20261111T012151/` and `chemprop_monroe_pca200_contended_20261111T013835/`. Contention distorts timing but not predictions, so those runs' 199,978-compound library predictions are valid and were retained.

The same guard caught one more run. `tabicl_monroe_pca62`'s first timing attempt (2026-11-18 08:29–08:39) recorded 15 intruder samples from a single foreign process present over 08:34–08:39 — roughly half the run — and was rejected unread as `tabicl_monroe_pca62_contended_20261118T091526/`. The published run is a clean repeat with 0 intruder samples. The contaminated figure was 557.2 s of model time against the clean 558.9 s, i.e. the contention cost nothing measurable here; the run was still discarded, because whether a contended measurement happened to be undisturbed is only knowable after a clean one exists. `tabicl_monroe_pca23` recorded 0 intruder samples on its first attempt and was not repeated.

The three 2,048-column sets (`tabicl_chemeleon`, `tabicl_morgan`, `tabicl_rdkit_morgan`) cost 102–115 h at the common chunk and 40–50 h tuned, and are not competitive on either axis at either setting.

`tabicl_monroe_pca100`'s best chunk is 50,000 (1.149 h) rather than 25,000 (1.447 h), matching `chemeleon_pca100` and not its own `pca200` sibling: the best chunk tracks column count more than it tracks the encoder.

## Effective dimensionality of the representations

How many directions does each representation actually use? Two estimators over the 7,032
aligned compounds, both unsupervised -- no model, no target, no splits, only the eigenvalue
spectrum. **Horn's parallel analysis** (q = 0.95, 100 simulations) is a hard count: retain
an observed eigenvalue only if it exceeds the 95th percentile of the same eigenvalue under
data with the correlation structure destroyed. **Participation ratio**, PR = (Σλ)² / Σλ², is
a soft one: it equals k for a spectrum with k equal eigenvalues and falls as the spectrum
concentrates.

| representation | full | non-constant | Horn's PA | PR (corr.) | 95% of variance | 99% |
|---|---:|---:|---:|---:|---:|---:|
| `monroe` | 720 | 720 | **62** | 31 | 74 | 160 |
| `chemeleon` | 2,048 | 1,941 | **169** | 70 | 442 | 1,033 |
| `rdkit` | 200 | 186 | **35** | 21 | 70 | 113 |

PA and PR are computed on the correlation matrix, the variance counts on the covariance
matrix, both over the non-constant columns: **107 of CheMeleon's 2,048 units are exactly
constant across all 7,032 compounds**, and 14 of the 200 RDKit descriptors are. rdkit also
contains a perfectly correlated descriptor pair (max |r| = 1.0000), so its correlation
matrix is singular and its rank is at most 185.

**All three use under a fifth of their nominal width, and the widest carries the least per
column.** CheMeleon holds the most structure in absolute terms (169 directions) but the
least per unit of width -- 8.7% of its non-constant columns, against Monroe's 8.6% and rdkit's
19%. Monroe reaches 62 real directions in 720 columns where CheMeleon needs 1,941 to reach
169, which is the same ranking as the models: on the identical splits, `tabicl_monroe` is
R² 0.627 against `tabicl_chemeleon`'s 0.477 (means over the 25 folds).

**This is the shape of the PCA results reported above.** Taking PR on the covariance
spectrum (what the centering-only PCA actually saw), CheMeleon's 27.7 falls to 26.3 at 720
components, 22.2 at 200 and 19.2 at 100 -- so the 1,328 columns dropped in the 2,048 → 720
step carried about 5% of its effective variance, which is why that step was a large *gain*
rather than a loss. Monroe's 22.6 falls to 22.4 at 200 components and 21.3 at 100: 99% and
94% retained, matching a projection that cost `tabicl_monroe` nothing that survives
test-compound resampling.

**But effective dimensionality is a property of the representation, not of its usefulness
to a given model.** 200 components is more than three times Monroe's 62 PA-significant
directions, and `chemprop_monroe_pca200` still lost 0.064 R² against `chemprop_monroe`.
What chemprop lost there is not PA-significant structure -- it is the unrotated basis. PA
and PR bound what *can* matter linearly; they do not predict what a particular consumer
needs.

### How these numbers were produced

- `all_batches/scripts/representation_dimensionality.py`, Pearson correlations throughout.
  The `shuffle` and `normal` randomizations give **identical** counts for all three
  representations, consistent with Dinno (2009) finding PA robust to the null's
  distributional form.
- Cross-checked against the `horns` package (0.6.0, the reference Python implementation of
  Horn's PA) on `monroe` at 50 simulations: **identical retained count, 62 vs 62**, with the
  threshold curves agreeing to a median of 0.5% (max 2.9%) over the first 124 positions,
  which is the region the decision rule can reach. horns returns its eigenvalues from
  `np.linalg.eigvals` without sorting, and the two curves do diverge by up to 77% -- at
  position 715, in the degenerate tail, where the rule has long since stopped.
- **rdkit's row is a Pearson number, and the estimator does move it a little.** horns would
  pick polychoric correlations for the 81 of rdkit's 186 non-constant columns that take 8 or fewer
  values; at full width that is 224 s per correlation matrix, about 1.3 h for 20 simulations,
  so Pearson is used throughout for comparability. Tested on the adversarial subset -- the
  40 lowest-cardinality non-constant columns, every one of them ordinal -- polychoric gives PA = 13
  against Pearson's 15 at 20 simulations, with scipy reporting non-convergence on some
  pairs. That is ~13% of the count, so rdkit's 35 would move by a few components under the
  other estimator, not by a factor. Monroe is untouched by the question -- its least varied
  column still takes 7,024 distinct values across 7,032 compounds, so horns would choose
  Pearson for all 720. CheMeleon has 14 non-constant columns that take 8 or fewer values (one takes
  only 2), so horns would use its mixed estimator there as well, on 14 columns of 1,941.
- **PA is not run on the PCA feature sets.** PCA scores are uncorrelated by construction
  (measured max |off-diagonal r| here is 0.033-0.042, non-zero only because the projection
  was fit on the 6,298 non-test rows and applied to all 7,032), so their correlation matrix
  is the identity and PA on it returns noise. Switching PA to the covariance spectrum does
  not help either: any null that preserves column variances reproduces a score matrix's
  spectrum exactly. The variants are checked the way that does work -- each one's covariance
  spectrum is the top-k truncation of its source's, to |Δλ|/λ₁ ≤ 3.0e-4.

## Notes

- TabICL was pre-trained on tables with 2–100 columns. The 2,048- and 2,248-column feature sets are outside that range and all score below the 200-column RDKit set; Monroe's native 720 is the best-performing width here despite also being outside it.
- The CheMeleon features are the pre-trained encoder outputs, not the 25 fine-tuned chemprop encoders.
- TabICL settings were not tuned (default `n_estimators`, normalization and feature shuffling).
- Both Random Forest models are untuned 100-tree baselines.
- The Random Forest models see train+val (6,298 rows); the TabICL models see the train partition only (~5,150 rows) as context. This favours the forests slightly and does not change the direction of any comparison above.
- 5 of the 7,032 PPB molecules have 2D rather than 3D Monroe input (MMFF/UFF unavailable or featurization timed out). They were kept, since dropping any row would change the 734-compound test set. All 7 of the extra molecules in chemprop's 7,039-row CSV embedded successfully, 5 of them on the same 2D fallback — they are the Se/As/Sn compounds that fail RDKit descriptors.
- The chemprop descriptor runs use chemprop's default descriptor scaling; no alternative was tried. `chemprop_monroe` adds 720 descriptors to a 300-d learned representation on 5,150 training rows, which is the configuration most at risk of overfitting, and is the reason the narrower `chemprop_monroe_pca200` was run alongside it. The result went the other way: the wide version is the one that works.

## ClearML

Project `ADMET/admet-model-training`. All TabICL tasks link input dataset `631c93d3e66d4878b7ced5a4cf19785d` (`.datasets/ppb_batches_1to5_testing_log10_unbound_tabicl`, parent: RF snapshot `cf132b4ceaaa4d2db1f31456bf010752`).

| Task (`ppb_batches_1to5_testing_log10_unbound_*_cv5x5`) | Task ID | Output model ID |
|---|---|---|
| `tabicl_rdkit` | `f5a5085b93f142768bb683fbbefeaef0` | `b32a4fb79f974f94ac488b7de9e3ffa7` |
| `tabicl_rdkit_morgan` | `306fdd5026ff45768794640de02d9f86` | `ffc3f895fb9a4fffa9316980f4a8a3cf` |
| `tabicl_chemeleon` | `ec65971fe63341f0a2f35698dd9ef942` | `7665b7927a434746b5433f4ee2d51d04` |
| `tabicl_morgan` | `140d099f87814737837c6597590d631e` | `1353b15ec0574e52a4ce47030fc79c8e` |
| `tabicl_chemeleon_pca720` | `3b244651dfe74c47a67198ae2e435be8` | `d0ff4d82790940078c5fc7c829075f5d` |
| `tabicl_chemeleon_pca200` | `11219f5a6a0549c09925473397c65284` | `aeda909a96af45f9adedde444305bb1e` |
| `tabicl_chemeleon_pca100` | `b54b900853004ea6b53bb61af8d03efe` | `5535420bb43e4e2085c52850ffc97cf0` |
| `tabicl_monroe` | `4f41feab2d594e50a44c390145b41217` | `fdbbf843c5cd44288d4b87bd33cfc8da` |
| `tabicl_monroe_pca200` | `64eed8ca3f68434b912fa81c7c9d1ce8` | `8d03c66c1eb94e82afc26ae85d50c856` |
| `tabicl_monroe_pca100` | `6822b66de8794f1296cbe35e5eb913f7` | `3522fa2eea54453c8667555617b59a93` |
| `tabicl_monroe_pca62` | `51a728d046894c7086e245754bb9019f` | `982331f3dc9b4b48bfe27a286c188fe1` |
| `tabicl_monroe_pca23` | `59df0d31c22c43fcb72c576285321873` | `1d814c07cecf4c1490c98b95675dba26` |

The two chemprop descriptor runs are registered by `train_chemprop_model_parallel.py` under the same project and link the chemprop input dataset `3a38a00b87e04038b360c7d3dc349eeb` (the one the baseline chemprop run used), not the TabICL dataset:

| Task | Task ID | Output model ID |
|---|---|---|
| `ppb_batches_1to5_testing_log10_unbound_chemprop_monroe_cv5x5` | `70f5b23302924ec3a63d7fc4774938d5` | `2775fe93eca84e45963ec01f91024547` |
| `ppb_batches_1to5_testing_log10_unbound_chemprop_monroe_pca200_cv5x5` | `4a54b95a6745469fa81c0e83eb13a676` | `638201e595324a7ba3f6f4723045735a` |

An earlier `chemprop_monroe` task exists from a launch whose 25 folds all failed with `ModuleNotFoundError: No module named 'rdkit'`: `chemprop_model.py` invokes the `chemprop` CLI as a subprocess, which resolved via PATH to `/home/andrew/.local/bin/chemprop` rather than the `proppredict-env` one. Its model directory was deleted and the log kept at `all_batches/models/logs/chemprop_monroe_pathfail.log`. The successful runs put the env's bin on PATH.

Comparison runs: chemprop `2d981b0b1f06417a8a7647ac80626017`, RF `fb3317344dd64a1697e640a382dd64ec`. The `chemprop_monroe` settings were read back from that chemprop task's own parameters rather than reconstructed, so data path, splits file, target, epochs, patience, dropout, foundation, worker count, GPU and dataset id match it exactly; the only differences are `--descriptors-path`, the model directory and the task name. The three `rf_monroe*` runs have no ClearML task — `train_ppb_rf_cv.py` writes fold predictions to disk and does not register to ClearML; their record is the model dirs `all_batches/models/ppb_batches_1to5_testing_log10_unbound_rf_monroe[_pcaK]/` and the log `all_batches/models/logs/monroe_rf.log`.

Code: proppredict branch `feature/tabicl-model`, commit `761f9fd` (`proppredict/models/tabicl_model.py`, `scripts/train_tabicl_model_cv.py`). The first four runs were made before that commit. The ClearML tasks record base commit `3591b466` plus the uncommitted diff of the 4 modified files (`models/__init__.py`, `training/__init__.py`, `training/trainer.py`, `setup.py`), which is identical to those files in `761f9fd`. The 2 new files (`tabicl_model.py`, `train_tabicl_model_cv.py`) were untracked at run time and are not in the ClearML diff. They were last modified at 10:39 PST, before the first run started (10:48 PST), and were committed unchanged in `761f9fd`. The `tabicl_morgan` run (2026-11-04) used proppredict branch `feature/tabicl-inference` at `b7d25ac` plus uncommitted inference additions to `tabicl_model.py` (the training path is unchanged), the `ppb-inference` conda env, and GPU 1 with no other process on it. The CheMeleon PCA and Monroe runs (2026-11-05) and the RF Monroe runs (2026-11-06) used the same branch and env on GPU 0.

The `tabicl_monroe_pca62` and `tabicl_monroe_pca23` runs (2026-11-16) used the same `ppb-inference` env on GPU 0, with the working tree at `b7d25ac` plus the uncommitted inference additions. Those additions are additive — a `--batch_size` passthrough in `train_tabicl_model_cv.py`, which was not passed, and inference methods appended to `tabicl_model.py` — so the training path is the code the earlier TabICL runs used, and the recorded estimator parameters match them exactly (`n_estimators` 8, `batch_size` 8, `random_state` 42, same checkpoint sha256). Two environment facts are needed to repeat any of these runs: `proppredict` is not installed in any conda env, so the script runs only with `PYTHONPATH=/home/andrew/projects/proppredict`; and `proppredict.models.__init__` imports `chemprop_model`, which imports `clearml` unconditionally, so `clearml` must be present even when `--log_to_clearml` is not passed. It was missing from `ppb-inference` on 2026-11-16 and was installed there (`clearml-2.2.0`, no other package added or upgraded).

Data and report scripts: `all_batches/scripts/build_tabicl_model_input.py`, `all_batches/scripts/build_chemeleon_pca_features.py`, `all_batches/scripts/embed_monroe.py`, `all_batches/scripts/train_ppb_rf_cv.py`, `all_batches/scripts/upload_assay_to_clearml.py --stage tabicl_model_input`, `analoging/scripts/assemble_ppb_tabicl_benchmark.py`. The 2026-11-09 chemprop runs added `--reuse_npz/--reuse_csv` to `embed_monroe.py` and `--project_with/--suffix` to `build_chemeleon_pca_features.py`; both exist so a grown row set reuses an existing embedding or projection instead of redrawing it.

## Verification

- Before: the six-model assembly (2026-11-04) gave ensemble R² 0.558 (chemprop), 0.556 (tabicl_rdkit), 0.499 (random_forest), 0.489 (tabicl_chemeleon), 0.483 (tabicl_rdkit_morgan), 0.246 (tabicl_morgan), all n = 734.
- After: the 15-model assembly (2026-11-06) gives 0.558, 0.556, 0.499, 0.489, 0.483 and 0.246 for the same six models, same n.
- After the 17-model assembly (2026-11-09): every one of the 15 pre-existing models is unchanged in n, MAE, RMSE, R², Pearson r and Spearman r to the full precision of `metrics_table.csv` (maximum absolute change 0.0), and n is still 734. `chemprop_monroe` and `chemprop_monroe_pca200` each contribute 735 test rows, which intersect to the same 734.
- After the 19-model assembly (2026-11-16): `metrics_table.csv` holds 19 rows, all at n = 734, and the six 2026-11-04 baseline models still give R² 0.558, 0.556, 0.499, 0.489, 0.483 and 0.246. The 2026-11-09 table was not kept, so this re-checks the published baseline values rather than diffing the previous file at full precision as the bullet above does.
- `monroe_parsed7039.npz` is (7039, 720) float32, all finite, no zero rows; its 7,032 rows that correspond to `aligned_parsed.csv` are bit-identical to `monroe.npz`, in that file's row order. `monroe_pca200_parsed7039.npz` is (7039, 200) and agrees with `monroe_pca200.npz` on those rows to 5.25e-06.
- Both chemprop runs report 25/25 folds, 0 failed folds, 25 `best.pt` and 25 `test_predictions.csv`, and log `Data alignment validated: 7039 samples, 720 features` (200 for the PCA run) with `train=5150, val=1154, test=735` on fold 1 — the same split sizes as the baseline chemprop run.

## Files

- `predictions/ppb.csv`: measured value, 25 fold columns per model (`<model>_model_<i>`), `<model>__mean` per model
- `metrics_table.csv`, `fold_metrics.csv`
- `tukey/`: ANOVA + Tukey HSD tables per metric, `summary_df.csv`
- `scatter_<model>.png`
- `tukey_intervals.png` — per-fold mean MAE, R² and Spearman r per model with Tukey HSD comparison intervals (FWER 0.05, blocked on CV fold: two-way ANOVA model + fold). Two models are distinguishable exactly when their whiskers do not overlap. Each model is coloured by its verdict against the best model on that metric: **blue = best, grey = tied (not significant), red = significantly worse**, following the presentation in Walters et al.'s benchmarking write-up, whose protocol — 5×5 CV replicates against one fixed held-out test set, compared by Tukey HSD — is the same one used here.
- `tukey_intervals_bootstrap.png` — the same chart, with "tied" decided by the paired test-compound bootstrap instead of Tukey, and each model's 95% bootstrap interval drawn as a pale bar. The two figures side by side are the caveat below, without needing a p-value: the tied-with-best set grows from 0/1/1 models (MAE/R²/Spearman) under Tukey to 2/3/3 under the bootstrap, and `chemprop_monroe` joins the tie with `tabicl_monroe` on both R² and Spearman.
- `representation_dimensionality_pa.png` — observed correlation-matrix eigenvalues and the 95% parallel-analysis threshold per representation, with the last retained component marked. `representation_dimensionality_pr.png` — participation ratio on the correlation and the covariance spectrum, with the 95%-of-variance count and the full column count. Both figures cover the three raw representations only: a PCA variant's correlation matrix is the identity, so neither quantity is defined for it.
- `representation_dimensionality.csv` (per representation: `p_nominal`, `p_live`, `pa_shuffle`, `pa_normal`, `pr_corr`, `pr_cov`, `n_var95_cov`, `n_var99_cov`, `max_abs_offdiag_r`), `representation_dimensionality_spectra.npz` (observed spectra and PA threshold curves, so the figure redraws without re-simulating) and `representation_dimensionality_crosscheck.json` (the comparison against the `horns` package).
- `tukey_intervals.csv` (means, `half_width`, `boot_lo/hi/se`, `p_vs_best`, `verdict`) and `tukey_intervals_pairs.csv` (all 171 pairs × 3 metrics with `p_tukey`, `p_boot_raw` and `p_boot_maxT`). Bootstrap B = 2,000 resamples, seed 0. These intervals are blocked on fold; the `tukey/` tables are not.
