"""Exploratory analysis of public-schema thermal-property data.

Publication-inspired descriptive diagnostics, not a replica of project data or a
means to reproduce the study's measured statistics. No EDA result is fed into
model fitting and no outlier is removed by the diagnostics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, TARGET, load_data, validate_data
from .reporting import environment, file_hash, save_json

VARIABLES = (*FEATURES, TARGET)
UNITS = {FEATURES[0]: "%", FEATURES[1]: "g/cm3", TARGET: "W/(m K)"}
TITLES = {
    FEATURES[0]: "Gravimetric water content (%)",
    FEATURES[1]: "Wet bulk density (g/cm³)",
    TARGET: "Thermal conductivity (W/(m K))",
}


def _validate_options(moisture_threshold, n_clusters, contamination, seed):
    if moisture_threshold is not None and (not isinstance(moisture_threshold, (int, float))
            or not np.isfinite(moisture_threshold) or moisture_threshold < 0):
        raise ValueError("moisture_threshold must be a finite nonnegative number or omitted")
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, int) or n_clusters < 2:
        raise ValueError("n_clusters must be an integer >= 2")
    if not isinstance(contamination, (int, float)) or not np.isfinite(contamination) or not 0 < contamination <= 0.5:
        raise ValueError("contamination must lie in (0, 0.5]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")


def descriptive_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe supplied *already validated* rows; use sample SD/skew/excess kurtosis.

    CV is SD / |mean|, undefined for zero mean. Pandas skew/kurt use adjusted
    sample formulas; small or constant groups report undefined measures as NaN.
    """
    rows = []
    for name in VARIABLES:
        x = pd.to_numeric(frame[name], errors="raise").astype(float)
        n = len(x)
        mean = float(x.mean()) if n else np.nan
        sd = float(x.std(ddof=1)) if n > 1 else np.nan
        q1 = float(x.quantile(0.25)) if n else np.nan
        q3 = float(x.quantile(0.75)) if n else np.nan
        rows.append({
            "variable": name, "unit": UNITS[name],
            "n": n, "mean": mean, "std_sample": sd, "min": float(x.min()) if n else np.nan,
            "q1": q1, "median": float(x.median()) if n else np.nan, "q3": q3,
            "max": float(x.max()) if n else np.nan, "iqr": q3-q1 if n else np.nan,
            "skewness_adjusted": float(x.skew()) if n >= 3 else np.nan,
            "kurtosis_excess_adjusted": float(x.kurt()) if n >= 4 else np.nan,
            "cv": sd/abs(mean) if n > 1 and abs(mean) > 1e-12 else np.nan,
        })
    return pd.DataFrame(rows)


def predictor_vif(frame: pd.DataFrame) -> pd.DataFrame:
    """VIF with a regression intercept (unlike uncentred VIF with no constant).

    Infinite or undefined values are represented as missing with a status; the
    function does not silently alter predictor columns or observations.
    """
    arr = frame[list(FEATURES)].to_numpy(dtype=float)
    rows = []
    for j, name in enumerate(FEATURES):
        row = {"predictor": name, "vif": np.nan, "r2_auxiliary": np.nan, "status": "ok"}
        if len(arr) < 3 or np.std(arr[:, j]) < 1e-12:
            row["status"] = "undefined_insufficient_or_constant_target"
        elif np.std(arr[:, 1-j]) < 1e-12:
            row["vif"], row["r2_auxiliary"] = 1.0, 0.0
        else:
            r2 = float(LinearRegression().fit(arr[:, [1-j]], arr[:, j]).score(arr[:, [1-j]], arr[:, j]))
            if r2 >= 1 - 1e-12:
                row["r2_auxiliary"], row["status"] = r2, "unbounded_perfect_collinearity"
            else:
                row["vif"], row["r2_auxiliary"] = 1.0 / (1.0 - r2), r2
        rows.append(row)
    return pd.DataFrame(rows)


def _pca_and_diagnostics(analysis: pd.DataFrame, *, n_clusters: int, contamination: float, seed: int):
    """Produce PCA, visual clusters and anomaly flags; skip ill-defined cases."""
    n = len(analysis)
    warnings = []
    scores = pd.DataFrame(columns=["record_id", "pc1", "pc2", "visual_cluster"])
    loadings = pd.DataFrame(columns=["variable", "pc1", "pc2"])
    variance = pd.DataFrame(columns=["component", "explained_variance_ratio", "eigenvalue"])
    flags = pd.DataFrame(columns=["record_id", "is_outlier", "isolation_score"])
    if n < 3:
        warnings.append("PCA, k-means and Isolation Forest skipped: fewer than three analysis records")
        return scores, loadings, variance, flags, warnings
    x = analysis[list(VARIABLES)].to_numpy(dtype=float)
    if (np.ptp(x, axis=0) <= 1e-12).any():
        warnings.append("PCA skipped: a variable is constant; no stable two-dimensional projection")
    else:
        scaled = StandardScaler().fit_transform(x)
        model = PCA(n_components=2)
        pcs = model.fit_transform(scaled)
        variance = pd.DataFrame({"component": ["PC1", "PC2"],
                                 "explained_variance_ratio": model.explained_variance_ratio_,
                                 "eigenvalue": model.explained_variance_})
        loadings = pd.DataFrame({"variable": VARIABLES, "pc1": model.components_[0], "pc2": model.components_[1]})
        scores = pd.DataFrame({"record_id": analysis.record_id.astype(str).to_numpy(),
                               "pc1": pcs[:, 0], "pc2": pcs[:, 1]})
        # Visualization only: clustering does not modify material classes or model inputs.
        unique = len(np.unique(np.round(pcs, decimals=10), axis=0))
        k = min(n_clusters, unique)
        if k >= 2:
            scores["visual_cluster"] = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(pcs)
            if k < n_clusters:
                warnings.append(f"Requested {n_clusters} clusters; only {k} distinct PCA points")
        else:
            scores["visual_cluster"] = -1
            warnings.append("Clustering skipped: fewer than two distinct PCA points")
    if n < 10:
        warnings.append("Isolation Forest skipped: fewer than ten analysis records")
    else:
        # Standardized diagnostics do not feed into training or remove any records.
        matrix = StandardScaler().fit_transform(x)
        iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=seed)
        raw_flags = iso.fit_predict(matrix)
        flags = pd.DataFrame({"record_id": analysis.record_id.astype(str).to_numpy(),
                              "is_outlier": raw_flags == -1,
                              "isolation_score": iso.decision_function(matrix)})
    return scores, loadings, variance, flags, warnings


def _save_plots(analysis: pd.DataFrame, scores: pd.DataFrame, correlation: pd.DataFrame,
                outdir: Path, dry: pd.DataFrame | None = None):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    folder = outdir / "plots"
    folder.mkdir(exist_ok=True)
    variables = list(VARIABLES)
    # Full pairplot analogue: marginal distribution + all bivariate relationships.
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    for i, y in enumerate(variables):
        for j, x in enumerate(variables):
            ax = axes[i, j]
            if i == j:
                ax.hist(analysis[x], bins="auto", edgecolor="white")
            else:
                ax.scatter(analysis[x], analysis[y], s=12, alpha=0.65, linewidths=0)
            if i == 2: ax.set_xlabel(TITLES[x], fontsize=8)
            if j == 0: ax.set_ylabel(TITLES[y], fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle("Exploratory pairplot (analysis subset)")
    fig.tight_layout()
    fig.savefig(folder / "pairplot.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    arr = correlation.to_numpy(dtype=float)
    image = ax.imshow(arr, cmap="coolwarm", vmin=-1, vmax=1)
    for i in range(3):
        for j in range(3):
            value = arr[i,j]
            ax.text(j, i, f"{value:.2f}" if np.isfinite(value) else "n/a",
                    ha="center", va="center", fontsize=10)
    ax.set_xticks(range(3), ["Moisture", "Wet density", "Conductivity"], rotation=30, ha="right")
    ax.set_yticks(range(3), ["Moisture", "Wet density", "Conductivity"])
    fig.colorbar(image, ax=ax, label="Pearson r", shrink=0.75)
    ax.set_title("Pearson correlations (analysis subset)")
    fig.tight_layout()
    fig.savefig(folder / "pearson_correlation.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, name in zip(axes, variables):
        ax.hist(analysis[name], bins="auto", edgecolor="white")
        ax.set_xlabel(TITLES[name], fontsize=9)
        ax.set_ylabel("Count")
    fig.suptitle("Marginal distributions (analysis subset)")
    fig.tight_layout()
    fig.savefig(folder / "distributions.png", dpi=160)
    plt.close(fig)

    if not scores.empty:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        classes = scores.visual_cluster.to_numpy()
        for k in np.unique(classes):
            m = classes == k
            ax.scatter(scores.loc[m,"pc1"], scores.loc[m,"pc2"], s=18, alpha=0.75,
                       label=f"Visual group {int(k)}")
        ax.set_xlabel("Principal component 1 (standardized variables)")
        ax.set_ylabel("Principal component 2")
        ax.set_title("PCA with illustrative k-means partition")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(folder / "pca_clusters.png", dpi=160)
        plt.close(fig)

    if dry is not None and not dry.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        for ax, name in zip(axes, variables):
            ax.hist(dry[name], bins="auto", edgecolor="white")
            ax.set_xlabel(TITLES[name], fontsize=9)
            ax.set_ylabel("Count")
        fig.suptitle("Below-threshold subset (descriptive only)")
        fig.tight_layout()
        fig.savefig(folder / "below_threshold_distributions.png", dpi=160)
        plt.close(fig)


def analyse_dataframe(frame: pd.DataFrame, output: str | Path, *, moisture_threshold=None,
                      n_clusters=3, contamination=0.02, seed=42, plots=True,
                      rejected_rows: pd.DataFrame | None = None, source_sha256: str | None = None):
    """Analyse validated rows and write transparent local CSV/PNG reports.

    Output should be a *new or empty* directory. This function will NOT modify
    its input frame or remove anomalies; nor does it change a fitted model.
    """
    _validate_options(moisture_threshold, n_clusters, contamination, seed)
    if set(VARIABLES + ("record_id",)) - set(frame.columns):
        raise ValueError("Expected validated public-schema data; required fields missing")
    if len(frame) < 3 or not np.isfinite(frame[list(VARIABLES)].to_numpy(dtype=float)).all():
        raise ValueError("EDA requires at least three finite, validated records")
    base = Path(output).resolve()
    if base.exists() and any(base.iterdir()):
        raise ValueError("EDA output directory must be new or empty")
    base.mkdir(parents=True, exist_ok=True)
    clean = frame.copy().reset_index(drop=True)
    below = None
    target = clean
    group_map = {"all_valid": clean}
    if moisture_threshold is not None:
        below = clean.loc[clean[FEATURES[0]] < moisture_threshold].copy()
        target = clean.loc[clean[FEATURES[0]] >= moisture_threshold].copy()
        group_map["below_threshold"] = below
        group_map["at_or_above_threshold"] = target
    if len(target) < 3:
        raise ValueError("EDA analysis subset requires >=3 rows after moisture threshold")
    descriptions = pd.concat([descriptive_statistics(df).assign(subset=key)
                              for key,df in group_map.items()], ignore_index=True)
    descriptions = descriptions[["subset", *[c for c in descriptions.columns if c != "subset"]]]
    descriptions.to_csv(base/"descriptive_statistics.csv", index=False)
    corr = target[list(VARIABLES)].corr(method="pearson")
    corr.to_csv(base/"pearson_correlation.csv")
    vif = predictor_vif(target)
    vif.to_csv(base/"predictor_vif.csv", index=False)
    scores, loadings, variance, flags, warnings = _pca_and_diagnostics(
        target, n_clusters=n_clusters, contamination=contamination, seed=seed)
    scores.to_csv(base/"pca_scores.csv", index=False)
    loadings.to_csv(base/"pca_loadings.csv", index=False)
    variance.to_csv(base/"pca_explained_variance.csv", index=False)
    flags.to_csv(base/"outlier_flags.csv", index=False)
    issues = (rejected_rows if rejected_rows is not None else
              pd.DataFrame(columns=["source_row", "reasons"]))
    issues.to_csv(base/"rejected_rows.csv", index=False)
    if plots:
        _save_plots(target, scores, corr, base, dry=below)
    report = {
        "software": "physical-consistency-toolkit 0.2.0",
        "analysis_type": "exploratory_diagnostics_only",
        "publication_results_reproduced": False,
        "source_sha256": source_sha256,
        "sample_sizes": {key: len(value) for key,value in group_map.items()},
        "n_rejected": len(issues),
        "moisture_threshold_pct": moisture_threshold,
        "analysis_subset": "at_or_above_threshold" if moisture_threshold is not None else "all_valid",
        "n_outliers_flagged": int(flags.is_outlier.sum()) if not flags.empty else None,
        "outliers_removed": False,
        "pca_components": 2 if not variance.empty else 0,
        "visual_clusters_requested": n_clusters,
        "outlier_contamination_configured": contamination,
        "random_seed": seed,
        "warnings": warnings,
        "environment": environment(),
    }
    save_json(base/"eda_report.json",report)
    lines=["# Exploratory data analysis (EDA)", "",
           "This is descriptive analysis, not a statistical or geological classification.",
           "PCA, k-means and anomaly flags are **diagnostic only** and do not modify ML features,"
           "the training split, or the fitted models. No records are removed as outliers.",
           "Generated figures and tables are from the user's dataset (or synthetic examples),",
           "not reproductions of published empirical figures or statistics.", "",
           f"Accepted valid rows: {len(clean)}; analysis subset: {len(target)}; rejected: {len(issues)}.",
           f"Moisture threshold: {moisture_threshold if moisture_threshold is not None else 'not applied'} %.",
           "", "Outputs: descriptive_statistics.csv (adjusted skew/excess kurtosis, CV and IQR),",
           "pearson_correlation.csv, predictor_vif.csv, pca_scores.csv, pca_loadings.csv,",
           "pca_explained_variance.csv, outlier_flags.csv, rejected_rows.csv, eda_report.json.",
           "Plots: pairplot, correlation heatmap, distributions, PCA clusters and optional dry-subset histograms.",
           "", "The sample CV is sample SD / |mean|; it is undefined for mean ~0.",
           "PCA is fitted to standardized (moisture, wet bulk density, conductivity).",
           "K-means clusters PC1/PC2 for visualization only; arbitrary group numbers are not lithological units.",
           "VIF is calculated using auxiliary linear regressions with an intercept.",
           "Isolation Forest flags are diagnostics; their nominal contamination is not an empirical error rate.",
           "", "Warnings:", *([f"- {s}" for s in warnings] if warnings else ["- None"]),
           "", "Local results may contain private user record identifiers: do not publish them unchecked."]
    (base/"EDA_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return report


def run_eda_file(input_path: str | Path, output_path: str | Path, *, column_map=None,
                 validation=None, moisture_threshold=None, n_clusters=3,
                 contamination=0.02, seed=42, plots=True):
    """External-user entry point: validate mapped data and run EDA on accepted rows."""
    source=Path(input_path).resolve()
    raw=load_data(source,column_map=column_map)
    rules=dict(validation or {})
    policy=rules.pop("on_invalid","error")
    valid,issues=validate_data(raw,on_invalid="drop",**rules)
    # Failure policy still exports explicit rejection reasons without running EDA.
    if len(issues) and policy=="error":
        out=Path(output_path).resolve()
        if out.exists() and any(out.iterdir()):
            raise ValueError("EDA output directory must be new or empty")
        out.mkdir(parents=True,exist_ok=True)
        issues.to_csv(out/"rejected_rows.csv",index=False)
        raise ValueError(f"Validation rejected {len(issues)} rows; see rejected_rows.csv")
    if policy not in ("error","drop"):
        raise ValueError("validation.on_invalid must be error or drop")
    return analyse_dataframe(valid,output_path,moisture_threshold=moisture_threshold,
                             n_clusters=n_clusters,contamination=contamination,
                             seed=seed,plots=plots,rejected_rows=issues,
                             source_sha256=file_hash(source))
