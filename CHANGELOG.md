# Changelog

## 0.2.0 - 2026-11-16

- Added separate, publication-inspired EDA CLI/API and optional integration into ML runs.
- Added descriptive statistics, Pearson correlation, VIF with intercept, standardized PCA, visualization-only k-means and non-destructive Isolation Forest flags.
- Added pairplot, distributions, heatmap and PCA scatter exports.
- Added independent synthetic dry/moist dataset with explicit generator and public TOML configuration.
- Added EDA unit/integration tests and updated documentation, metadata and release files.
- Existing PCI equations, model families and post-hoc comparison behavior preserved.
- No original project records, original table schemas, fitted estimators or legacy code shipped.
- EDA on synthetic data does not reproduce the published empirical measurements or results.

## 0.1.0 - 2026-11-06

- New modular public implementation of regression, surfaces and PCI.
- S1-S4 diagnostics, shared maximum-roughness normalization and sum-to-one weights.
- Explicit coordinate-aware first-order differences and masked-domain extension.
- Independent synthetic examples, analytical tests and end-to-end offline checks.
- Training-only scaling/CV; optional specimen grouping; external-surface mode.
- CSV/XLSX, TOML, Python API, CLI, Spyder runner and reports/plots.
- Optional XGBoost and self-contained interactive HTML exports.
- No historical executable, private schema adapter, data or trained model included.
- Empirical reproduction is not established by synthetic tests.
- Authorship and ownership confirmed by Mateusz Żeruń on 2026-11-06: code
  developed for his cycle of doctoral publications; MIT license confirmed.
- Updated publication metadata and rights status; numerical code is unchanged.
