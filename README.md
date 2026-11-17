# Physical Consistency Toolkit

Version 0.2.0 - EDA-integrated release prepared 16 November 2026.

A small Python package for exploratory data analysis (EDA), comparing thermal-conductivity regressors and inspecting
physical consistency of their two-predictor response surfaces. It also accepts
surfaces from external models without retraining. All bundled data are independently
constructed **synthetic examples**. There are no laboratory records, fitted model
files, article figures, or private data adapters in this release.

**Author and copyright holder:** Mateusz Żeruń. **License:** MIT.
The author confirmed on 6 November 2026 that the code was developed for his
cycle of doctoral publications and that he holds the rights. See NOTICE.md.
PIG-PIB is listed solely as scientific affiliation.

## Install

Use Python 3.11 or newer. Python 3.11.5 / Windows / Anaconda / Spyder is the target;
actual test environment and unverified platforms are recorded in VALIDATION.md.
Extract the source package, open Anaconda Prompt in its root and run:

```bash
python -m pip install .
python -m physical_consistency demo --output my_synthetic_demo
```

The demo generates 240 artificial records, creates an EDA report and analyses SVR, Random Forest,
Decision Tree and k-Nearest Neighbours. It requires no network **after dependencies
are installed**. View `my_synthetic_demo/results/report.md` and the `plots` folder.
The demonstration refuses to overwrite a nonempty result directory.

Optional features:

```bash
python -m pip install ".[xlsx]"         # XLSX input
python -m pip install ".[xgboost]"      # fifth model family
python -m pip install ".[interactive]"  # self-contained offline 3D HTML
```

Set `interactive = true` for HTML surfaces; Matplotlib PNG surfaces are available
without Plotly. Set `plots = false` for numerical analysis only.

## Use your own data

Create a UTF-8 CSV with these columns (one measurement per row):

| Column | Meaning | Unit |
|---|---|---|
| record_id | Unique user-assigned identifier | text |
| moisture_mass_pct | Gravimetric water content, water mass / dry solid mass x 100 | percent |
| bulk_density_g_cm3 | Wet bulk mass / total specimen volume | g/cm3 |
| thermal_conductivity_w_mk | Observed thermal conductivity | W/(m K) |
| material_group | Optional single-material label or explicit filter | text |
| specimen_id | Optional grouping of repeated states of the same specimen | text |

Units are declared, not inferred from the numeric values. Convert dry bulk density,
kg/m3 or fractional water contents yourself before use. See docs/data_schema.md.
Copy `examples/demo.toml`, set `input`, `output` and material/validation settings,
then run:

```bash
python -m physical_consistency validate --input my_data.csv --report issues.csv
python -m physical_consistency run --config my_analysis.toml
```

All input/output paths inside TOML are relative to that file. With Spyder, run
`examples/run_spyder.py` after installing in Spyder's interpreter. See docs/usage.md
for custom columns, repeated specimens, hyperparameter tuning and optional XGBoost.

## Exploratory Data Analysis (EDA)

The **EDA** command analyses public-schema CSV/XLSX before any model training. It
produces descriptive statistics (N, sample SD, range, quantiles, median, IQR,
adjusted skewness and excess kurtosis, coefficient of variation), Pearson
correlations, intercept-corrected predictor VIF, standardized 3-variable PCA,
illustrative k-means clusters in PC1–PC2 space and diagnostic Isolation Forest
flags. Plots include a pairplot, marginal histograms, Pearson heatmap and PCA
scatter. **Neither outliers nor clusters are used to remove or classify records.**

Run the independent example with newly generated dry/moist *synthetic* states:

```bash
python -m physical_consistency eda --config examples/eda_demo.toml
```

Or analyse your own validated table:

```bash
python -m physical_consistency eda --input my_data.csv --output my_eda
python -m physical_consistency eda --input my_data.csv --output my_eda_split --moisture-threshold 1.0
```

The dry/moist threshold is **optional**, not a universal filtering rule.
Below-threshold states are summarized and plotted but excluded only from the
PCA/correlation subset when the threshold is explicitly requested. Running EDA
never changes the training or evaluation samples. For EDA together with model
fitting, set `[eda] enabled = true` in your TOML. Read **docs/eda.md** for details,
limits and safe handling of your local outputs.

## Evaluate existing response surfaces

```bash
python -m physical_consistency evaluate-surface --input examples/synthetic_surfaces.csv --output surface_review
```

Provide all models in the same long-form CSV with `model_id`, the two predictor
columns, `prediction_w_mk`, and `inside_domain`. Include every rectangular lattice
coordinate and explicitly mask excluded nodes. The package refuses duplicate
coordinates, incomplete grids and mismatched comparison domains. The Python API
also accepts any external model with `predict(X)`; see examples/external_model.py.

## Interpretation

The default PCI follows Supplement S2, equations S1-S4, of the methodological paper:

`PCI = 0.4 MVR + 0.4 Rough_norm + 0.2 PBV`, with operational bounds 0.2-5 W/(m K).

Lower is better only within a fixed comparison cohort, domain, units, grid and
criteria. Rough_norm uses **the maximum raw roughness among all compared models**.
It penalises first-derivative magnitude, so a steep smooth plane can receive a
larger penalty than a gently sloping surface. PCI is a post-hoc diagnostic and is
never a training objective. It does not prove physical validity.

The equations are implemented with explicit numerical choices: forward differences,
backward boundary fallback and no undocumented smoothing. Published documents do
not identify every historical numerical setting. This package is an independent
implementation, **not a verified reproduction of the paper's empirical PCI values**.
Hull masks, altered criteria and new weight variants are documented extensions.

Read docs/mathematics.md, docs/eda.md, docs/usage.md, docs/api.md, docs/synthetic.md, VALIDATION.md
and docs/release_status.md before interpreting scientific results.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests distinguish analytical S1-S4 checks from workflow checks. They require no
private data. Optional extension tests explicitly skip if their library is missing.

## Citation and authorship

Software: Mateusz Żeruń. Physical Consistency Toolkit, version 0.2.0 (2026).
Use CITATION.cff. No software DOI has been assigned.

Methodological reference: Żeruń, M., Jagoda, E., and Majer, E. (2026).
Machine-Learning-Based Parameterisation of Soil Thermal Conductivity for Shallow
Geothermal and Ground Heat Exchanger Modelling. Energies 19(8), 1827.
https://doi.org/10.3390/en19081827

Affiliation: Polish Geological Institute - National Research Institute (PIG-PIB),
scientific affiliation only. It does not assign copyright or imply endorsement.
