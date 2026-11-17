# Installation and operation

## Anaconda and Spyder

In Anaconda Prompt activate the environment used by Spyder, change to the extracted
package root and run `python -m pip install .`. The Python interpreter selected in
Spyder must belong to that environment. Restart its kernel after installation.
The target is Python 3.11.5; the run actually tested is stated in VALIDATION.md.

Run examples/run_spyder.py using Run file. It locates examples/demo.toml via
__file__ and works independently of Spyder's working directory. Change output in
the TOML before rerunning because nonempty output folders are protected.

A direct Python entry point is also available:

```python
from physical_consistency import run_analysis
summary = run_analysis("C:/research/my_analysis.toml")
```

Use forward slashes or TOML literal strings for Windows paths. Relative TOML paths
are resolved against the configuration file. A dictionary config can instead be
passed with base_dir='C:/research'. No private infrastructure is needed.

## Custom CSV columns and filtering

Use a table like this in your own configuration (these names are fictitious):

```toml
input = "measurements.csv"
output = "analysis_01"

[column_map]
reading_key = "record_id"
water_percent = "moisture_mass_pct"
wet_density = "bulk_density_g_cm3"
measured_lambda = "thermal_conductivity_w_mk"

[validation]
on_invalid = "drop"
min_moisture = 1.0
material_group = "my_selected_material"

[validation.ranges]
moisture_mass_pct = [0.0, 60.0]
bulk_density_g_cm3 = [0.8, 3.0]
thermal_conductivity_w_mk = [0.01, 12.0]
```

The material column must be supplied if material_group filtering is selected.
Missing material labels cannot be treated as a valid material. Moisture >=1% is
an explicit optional filter, not an automatic universal assumption.

Validate using your complete settings:
`python -m physical_consistency validate --input measurements.csv --config my_analysis.toml --report issues.csv`.
Exit code 0 means no rejected records, 1 means row issues, 2 means an error.
Review the report before accepting any exclusions. Outputs from your own data may
contain confidential values; review them before sharing.

## Splits, models and tuning

Default outer split is 80/20, seed 42. Scaling for SVR and KNN is fitted only on
training rows, inside pipelines. Trees use unscaled predictors. Automatic tuning
is off by default; examples/tuned.toml illustrates training-only GridSearchCV
optimising R2. All selected models share an identical split. Baseline hyperparameters
are illustrative toolkit defaults, not a replication of the published study.

Use split='group' and specimen_id for repeated measurements of a specimen. The
holdout and inner CV keep these groups separate. Inspect group sizes and the
split_assignments.csv file. There is no stratification or inference of sample
independence from coordinates. Model comparison against the same holdout should
not turn into repeated adaptive test-set tuning.

Each [[models]] table has id, family, optional params, optional search_grid and
cv_folds. Native estimator parameter names are used, without pipeline prefixes:

```toml
[[models]]
id = "SVR_regularised"
family = "svr"
cv_folds = 5
[models.search_grid]
C = [1.0, 10.0]
epsilon = [0.03, 0.1]

[[models]]
id = "XGB"
family = "xgb"
[models.params]
n_estimators = 100
max_depth = 3
learning_rate = 0.05
```

Supported families: svr, rf, dt, knn; xgb requires [xgboost]. No missing optional
model is silently substituted. Keep KNN neighbours below the size of every inner
training fold. Unknown configuration keys fail with a message.

## Domain and PCI configuration

```toml
[grid]
shape = [61, 61]
domain = "hull"
# Optional explicit ranges; otherwise training min/max is used.
# moisture_range = [2.0, 17.0]
# density_range = [1.35, 1.75]

[pci]
weights = [0.4, 0.4, 0.2]
bounds = [0.2, 5.0]
directions = [1, 1]
tolerance = 0.0
[pci.weight_variants]
my_sensitivity = [0.25, 0.55, 0.2]
```

Use domain='rectangle' only with awareness that it can include extrapolation.
Default hull support uses training predictors only; withheld targets never enter
model fitting, scaling, tuning, or PCI calculations. The default grid is 41x41,
a toolkit choice. Compare several resolutions, particularly for tree models.

## Imported surfaces

`python -m physical_consistency evaluate-surface --input all_surfaces.csv --output evaluation --config my_analysis.toml`
uses the configuration's PCI and plot settings. No fitting, RMSE or Pareto claim
is made from surfaces alone. Supply all comparator surfaces in one CSV so
Rough_ref is shared. Examples include constant, increasing plane, decreasing
plane, checkerboard, oscillatory and out-of-bounds surfaces on one grid.

## Output files

| File | Contents |
|---|---|
| metrics.csv | accuracy, PCI components, common Rough_ref and Pareto flag |
| weight_sensitivity.csv | all weight variants on identical surfaces; Pareto flags when accuracy exists |
| surfaces.csv | predictions and masks on a regular lattice |
| surface_diagnostics.csv | per-node derivatives, eligibility and violations |
| holdout_predictions.csv | observed, predicted and residual (prediction - observation) |
| split_assignments.csv | measurement ID, train/test and optional specimen ID |
| rejected_rows.csv | rejected table row number and all reasons |
| config.json | fully resolved run settings |
| run_metadata.json | units, axes, input hash, seeds/settings and dependency versions |
| report.md | result table and interpretation |
| plots/ | 2D, 3D, holdout diagnostics, RMSE-PCI and component comparison PNGs |

No trained estimator is pickled. Runs refuse to overwrite nonempty outputs. If a
run fails, its directory may contain a partial report; inspect the error and use a
new output folder for retry. The package never sends analysis data over a network.

## Classical physical models

Historical functions carrying names of established physical models were simplified
or ambiguous; none is silently distributed under those names. Verified external
physical models can be evaluated through predict(X) or imported surfaces. This
release focuses on regression and PCI; it makes no new claim to implement Kersten,
Johansen or Lu formulations.


## Run EDA independently or during fitting

EDA reads the same public schema and optional source-to-public `column_map` as
modelling. It does not require a particular file name, lithology, project code or
private database schema.

```bash
python -m physical_consistency eda --input examples/synthetic_eda.csv --output eda_example --moisture-threshold 1
python -m physical_consistency eda --config examples/eda_demo.toml
```

`--config` resolves the input/output paths relative to the TOML file. Command
line `--input`/`--output` paths are resolved against your terminal directory.
Outputs must be new or empty. In a TOML run, add:

```toml
[eda]
enabled = true
moisture_threshold = 1.0   # optional; omit to analyse all valid rows
n_clusters = 3
contamination = 0.02
plots = true
```

EDA under `run` writes to `results/eda` and includes validated below-threshold
observations in the descriptive table **even if** the modelling filter
`validation.min_moisture` excludes them. To deliberately drop such samples
from modelling, set `[validation] on_invalid = "drop"`; the rejected-record
log records the exclusions. EDA never changes that policy, leaks statistics
into fitting, or performs automatic outlier filtering.

Standalone EDA uses `validation.ranges` and `validation.material_group` from the
TOML, but excludes `validation.min_moisture`: the EDA-specific threshold defines
which states are described as below threshold. PCA and Pearson correlations
are calculated only on the at-or-above-threshold subset, where configured.

See docs/eda.md for the definitions, file outputs, warnings and limitations.
