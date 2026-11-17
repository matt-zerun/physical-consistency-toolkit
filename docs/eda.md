# Exploratory Data Analysis (EDA)

**Version 0.2.0 | Synthetic demonstrations and a public data contract**

This module implements the *types* of exploratory analyses documented in
Żeruń et al., Energies **19** (2026), 1827, Sections 2.3–2.4 and Supplement S1:
a moist-state pairplot (Fig. 1), Pearson correlations (Fig. 2), PCA and k-means
visualization (Fig. 3), and descriptive statistics for global, below-threshold
and above-threshold subsets (Table 1). It also offers extra diagnostics (VIF,
Isolation Forest) found in the legacy exploratory code but not required for the
publication figures. These are **new independent implementations**, not validated
exact reproductions of the published source data, outputs or figure aesthetics.

## Commands

```bash
python -m pip install .
python -m physical_consistency eda --config examples/eda_demo.toml
python -m physical_consistency eda --input examples/synthetic_eda.csv --output my_eda --moisture-threshold 1
```

The threshold is user-selected. If omitted, analysis covers all validated rows.
When provided, three descriptive subsets are written: `all_valid`,
`below_threshold` and `at_or_above_threshold`. Multivariate diagnostics use
only the at-or-above-threshold set; below-threshold observations are described
but not automatically removed from your source file. The 1% criterion was a
study-specific decision for the article, not a universal geological threshold.

## Output tables

| File | Meaning |
|---|---|
| `descriptive_statistics.csv` | Count, mean, sample SD, min, Q1, median, Q3, max, IQR, adjusted sample skewness and excess kurtosis, CV=SD/absolute mean |
| `pearson_correlation.csv` | Pearson coefficients for water content, wet bulk density, and conductivity in the analysis subset |
| `predictor_vif.csv` | Predictor-only VIF using auxiliary linear regressions **with intercept**; perfect collinearity reported, not coerced to a finite value |
| `pca_scores.csv` | PC1, PC2 and k-means visualization label per user-supplied record ID |
| `pca_loadings.csv` | PC component directions for standardized water content, wet bulk density and conductivity |
| `pca_explained_variance.csv` | Explained ratios and eigenvalues of PCs 1 and 2 |
| `outlier_flags.csv` | Isolation Forest anomaly flags and decision scores; observations **are not dropped** |
| `rejected_rows.csv` | Validation issues; no source-data values included |
| `eda_report.json` | Counts, thresholds, diagnostics, environment and warnings |
| `EDA_REPORT.md` | Summary, methodological warnings and disclosure |

Optional plots are `pairplot.png`, `distributions.png`, `pearson_correlation.png`,
`pca_clusters.png` and `below_threshold_distributions.png` when there are
below-threshold records. All plots use generated **local** results; no published
article figure or actual observation is embedded in the repository.

## Methods and limits

- Descriptive skewness and kurtosis follow pandas adjusted sample estimators;
  kurtosis is *excess* kurtosis. SD uses ddof=1, IQR=Q3−Q1 and CV=SD/|mean|.
  CV is undefined when mean≈0; skewness and kurtosis require sufficient data.
- Pearson r measures **linear** dependence, not causal dependence or physical
  validity. Constant input variables can produce undefined correlations.
- PCA uses standardized three-variable input (including conductivity) for
  exploration only; no PCA scores enter subsequent ML training. Two components
  are retained when the data are nondegenerate.
- k-means uses fixed seed and explicit n_init=10. Clusters partition a
  continuous numerical space; they are **not** a geological classification.
- VIF is based on an intercept-corrected auxiliary regression on the two
  model predictors; infinite/undefined cases have explicit status strings.
- Isolation Forest with configured contamination returns **flags**, not a
  definitive list of erroneous measurements or an instruction to exclude data.
  All flags are preserved in a separate table; none affect training.
- Running EDA on all available data is acceptable for descriptive *post hoc*
  inspection. If exploratory conclusions are used for feature/parameter tuning,
  perform those selections inside training folds to avoid information leakage.
- Any model/data-driven assertion (monotonicity, no classes, low collinearity)
  requires the user's own data; this module does not automatically assert the
  original article's empirical findings for new datasets.

## Privacy and provenance

Input/output are controlled by the user, remain local, and may contain user
record identifiers. **Do not publish real local outputs inadvertently.** The
repository includes only independently created synthetic records. Public column
names do not reproduce a private project schema; arbitrary external source
columns may be mapped with TOML `column_map`.

## Standalone EDA in Spyder

After installing the package in Spyder's interpreter, open
`examples/run_eda_spyder.py` and execute the file. It locates synthetic data
relative to `__file__` rather than relying on the current working directory.
Use a new directory for a second run; existing nonempty folders cannot be
silently overwritten.
