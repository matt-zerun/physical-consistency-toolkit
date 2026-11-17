# Python API

```python
from physical_consistency import (
    generate_synthetic, load_data, validate_data, generate_grid, Surface,
    write_surfaces, read_surfaces, components, compare_surfaces, run_analysis,
    run_eda_file, analyse_dataframe, descriptive_statistics, predictor_vif,
    generate_synthetic_eda,
)
```

| Function | Inputs | Output |
|---|---|---|
| generate_synthetic_eda(n_moist=180, n_dry=45, seed=64021) | independent illustrative dry/moist data | synthetic pandas DataFrame |
| descriptive_statistics(validated_frame) | public 3-variable table | descriptive table including skewness, excess kurtosis, IQR, CV |
| predictor_vif(validated_frame) | public table | VIF with intercept and status flags |
| analyse_dataframe(validated_frame, output, moisture_threshold=None, ...) | previously validated frame | EDA report dictionary; writes local tables/plots |
| run_eda_file(input_path, output_path, column_map=None, validation=None, ...) | CSV/XLSX, public mapping, explicit validation options | EDA report dictionary; writes local tables/plots |
| generate_synthetic(n=240, seed=73021) | independent sample size and seed | synthetic pandas DataFrame |
| load_data(path, column_map=None) | CSV/XLSX and optional source-to-public names | raw DataFrame |
| validate_data(frame, ranges=None, on_invalid='error', min_moisture=None, material_group=None) | public table and explicit filters (options are keyword-only) | accepted DataFrame, rejection DataFrame |
| generate_grid(moisture_range, density_range, shape=(41,41), support=None) | axes/ranges and optional (n,2) hull support | Surface template with zero included values |
| Surface(moisture, density, z, mask=None) | axes, (n_w,n_r) prediction array, optional boolean mask | validated surface |
| components(surface, bounds=(0.2,5), directions=(1,1), tolerance=0) | one response surface | raw diagnostic dictionary; no normalised PCI |
| compare_surfaces(surfaces, weights=(0.4,0.4,0.2), **criteria) | model-id to Surface mapping | DataFrame with joint normalisation and PCI |
| write_surfaces(surfaces, path) | mapping and CSV target | writes public long-form table |
| read_surfaces(path) | public surface CSV | model-id to Surface mapping |
| run_analysis(config, base_dir=None) | TOML filename or dictionary | DataFrame; writes reports |

Additional modules: physical_consistency.models.make_model and fit_model;
physical_consistency.pci.gradients; physical_consistency.evaluation.accuracy and
pareto_mask; physical_consistency.workflow.evaluate_surface_file.
Estimator factories return native scikit-learn-compatible objects. New estimator
families can be used via the external-model API without modifying the toolkit.

Example with analytical external surfaces:

```python
import numpy as np
from physical_consistency import Surface, generate_grid, compare_surfaces
base = generate_grid((2, 17), (1.35, 1.75), shape=(41,41))
w, d = np.meshgrid(base.moisture, base.density, indexing='ij')
models = {'plane': Surface(base.moisture, base.density, 0.5+0.05*w+0.3*d),
          'constant': Surface(base.moisture, base.density, np.ones_like(w))}
print(compare_surfaces(models))
```

Coordinates, units and domain masks must match exactly across a comparison.
Surface axes and arrays are copied and validated. There is no executable model
loading from untrusted files and no network dependency at analysis time.
