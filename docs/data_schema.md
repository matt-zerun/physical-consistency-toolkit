# Public data contract

A new flat table stores independent measurements. No internal relational keys,
project categories or private column adapters are bundled. Required columns are
record_id, moisture_mass_pct, bulk_density_g_cm3, thermal_conductivity_w_mk.
Extra user columns are ignored for fitting. material_group, specimen_id and
is_synthetic have optional explicit meanings described below.

| Field | Type | Definition |
|---|---|---|
| record_id | nonempty unique text | user-assigned measurement key |
| moisture_mass_pct | finite number | 100 x mass of water / mass of dry solid |
| bulk_density_g_cm3 | finite number | wet total mass / total specimen volume |
| thermal_conductivity_w_mk | finite positive number by default | observed lambda |
| material_group | optional nonempty text | one material or an explicitly selected subset |
| specimen_id | optional nonempty text | repeated states sharing an original specimen |
| is_synthetic | optional boolean | provenance marker; not a guarantee about user data |

Default plausibility ranges are broad software validation choices: moisture 0-100%,
wet bulk density 0.1-4 g/cm3, conductivity 0.001-20 W/(m K). They are not the PCI
bounds or empirical quantiles. Ranges can be overridden explicitly in TOML.
Numbers alone cannot reveal wrong units. No unit inference or automatic conversion
is performed; bulk and dry densities must not be interchanged. For reference,
wet density = dry density x (1+w/100) with gravimetric w, if consistent measurements
and definitions justify conversion. Conversion must be performed by the user.

Missing/nonnumeric/infinite numeric fields, implausible ranges, blank IDs, all
occurrences of duplicate IDs and all duplicate predictor-target triplets are
flagged. Record IDs are compared exactly; users should normalise their own IDs.
Duplicate predictor coordinates with different measured targets may represent
replicates; supply specimen_id when related measurements must stay in one split.
The toolkit cannot infer unlabelled relationships between observations.

Default invalid-record handling is error. The validate command creates a report;
run also writes rejected_rows.csv before stopping on row errors. An explicit
on_invalid='drop' rejects flagged rows with source-row/reason reporting. Nothing
is imputed, clipped, repaired or rejected using a fitted outlier model. Source_row
means tabular record position plus one header, not physical line number for CSV
fields containing newlines. Group-selection and moisture-threshold exclusions
also appear in the same report.

Multiple material groups must be explicitly reduced to one. Repeated specimen_id
requires split='group'; inner CV uses GroupKFold, outer splitting GroupShuffleSplit.
Use representative groups and inspect their sizes. Random splits assume independent
rows. Input/output directories are user-controlled and not auto-published.

## Surface exchange

One long-form CSV contains model_id, moisture_mass_pct, bulk_density_g_cm3,
prediction_w_mk, inside_domain. model_id is a short ASCII safe identifier starting
with a letter. All rectangular lattice coordinates must occur exactly once per
model. inside_domain is true/false or 1/0. Values outside the mask may be blank;
inside values must be finite. Axes are regular, finite, strictly increasing after
sorting. All compared models must have exactly the same axes and mask. No pivot
aggregation, interpolation or NaN filling is performed. Shuffling rows is allowed.

Surface metadata assumes the same predictor and target units as the data table.
Supply scientifically justified common grids/masks; sampling each model on its own
domain prevents a meaningful joint PCI comparison and is rejected.


## EDA and threshold selection

EDA uses exactly the same public measurement columns and requires no additional
private/project fields. The optional `moisture_threshold` is for descriptive
subsetting (below vs at-or-above); it is not a universal dryness definition and
does not silently remove observations. An Isolation Forest flag is not an
automatic exclusion. Local EDA exports include the supplied `record_id`, so
review before sharing output derived from your own records.
