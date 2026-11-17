"""Public flat-table schema and explicit record-level validation."""
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ["moisture_mass_pct", "bulk_density_g_cm3"]
TARGET = "thermal_conductivity_w_mk"
REQUIRED = ["record_id", *FEATURES, TARGET]
DEFAULT_RANGES = {FEATURES[0]: [0, 100], FEATURES[1]: [0.1, 4], TARGET: [0.001, 20]}


def load_data(path, column_map=None):
    """Read CSV or optional XLSX. Map user column names to public column names."""
    p = Path(path)
    text_types={"record_id":"string","specimen_id":"string"}
    for source,target in (column_map or {}).items():
        if target in text_types:
            text_types[source]="string"
    if p.suffix.lower() == ".csv":
        frame = pd.read_csv(p, dtype=text_types)
    elif p.suffix.lower() == ".xlsx":
        try:
            frame = pd.read_excel(p, engine="openpyxl", dtype=text_types)
        except ImportError as exc:
            raise ValueError("XLSX requires installation with the [xlsx] extra") from exc
    else:
        raise ValueError("Input must be .csv or .xlsx")
    if column_map:
        if not isinstance(column_map, dict) or not set(column_map).issubset(frame.columns):
            raise ValueError("column_map must map existing user columns to public names")
        frame = frame.rename(columns=column_map)
    if frame.columns.duplicated().any():
        raise ValueError("Column mapping produces duplicate columns")
    return frame


def validate_data(frame, *, ranges=None, on_invalid="error", min_moisture=None,
                  material_group=None):
    """Return (accepted, rejection_report). Never impute or silently remove rows.

    Rejection rows contain a source row number and reasons, not original records.
    Default policy raises; select on_invalid='drop' explicitly to reject rows.
    All occurrences of duplicate IDs and duplicate numeric triplets are flagged.
    """
    if on_invalid not in ("error", "drop"):
        raise ValueError("on_invalid must be 'error' or 'drop'")
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required public columns: {sorted(missing)}")
    f = frame.copy().reset_index(drop=True)
    reasons = [[] for _ in range(len(f))]
    def flag(mask, reason):
        for i in np.flatnonzero(np.asarray(mask, dtype=bool)):
            reasons[i].append(reason)
    ids = f.record_id.astype("string")
    flag(ids.isna() | ids.str.strip().eq("").fillna(True), "missing_record_id")
    flag(ids.duplicated(keep=False), "duplicate_record_id")
    limits = dict(DEFAULT_RANGES)
    if ranges:
        if set(ranges)-set(limits):
            raise ValueError("Unknown validation range field")
        limits.update(ranges)
    for col in [*FEATURES, TARGET]:
        values = pd.to_numeric(f[col], errors="coerce")
        flag(~np.isfinite(values), f"nonfinite_or_nonnumeric:{col}")
        lo, hi = limits[col]
        if not np.isfinite([lo,hi]).all() or lo >= hi:
            raise ValueError(f"Invalid validation range: {col}")
        flag((values < lo) | (values > hi), f"outside_validation_range:{col}")
        f[col] = values
    flag(f.duplicated(subset=[*FEATURES, TARGET], keep=False), "duplicate_measurement_triplet")
    if min_moisture is not None:
        if not np.isfinite(min_moisture):
            raise ValueError("min_moisture must be finite")
        flag(f[FEATURES[0]] < min_moisture, "below_selected_moisture_threshold")
    if material_group is not None:
        if "material_group" not in f:
            raise ValueError("material_group selection requires that column")
        flag(f.material_group != material_group, "outside_selected_material_group")
    if "specimen_id" in f:
        v = f.specimen_id.astype("string")
        flag(v.isna() | v.str.strip().eq("").fillna(True), "missing_specimen_id")
    rejected = pd.DataFrame([{"source_row":i+2,"reasons":";".join(r)}
                             for i,r in enumerate(reasons) if r],
                            columns=["source_row","reasons"])
    keep = np.array([not r for r in reasons], dtype=bool)
    accepted = f.loc[keep].copy()
    if on_invalid == "error" and len(rejected):
        raise ValueError(f"Validation rejected {len(rejected)} of {len(f)} rows; "
                         "run validate to obtain a report or select on_invalid='drop'")
    if "material_group" in accepted:
        if (accepted.material_group.isna() | accepted.material_group.astype(str).str.strip().eq("")).any():
            raise ValueError("material_group cannot be missing")
        if accepted.material_group.nunique() > 1:
            raise ValueError("Select one material_group before fitting a two-predictor model")
    return accepted, rejected
