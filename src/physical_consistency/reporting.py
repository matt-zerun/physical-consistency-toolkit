"""Portable run provenance and Markdown reporting without serialised estimators."""
import hashlib
import json
from pathlib import Path
import platform
from importlib.metadata import version,PackageNotFoundError


def save_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")


def environment():
    result={"python":platform.python_version(),"system":platform.system()}
    for name in ["numpy","pandas","scipy","scikit-learn","matplotlib","openpyxl","xgboost","plotly"]:
        try:
            result[name]=version(name)
        except PackageNotFoundError:
            result[name]=None
    return result


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_report(output,table,metadata):
    lines=["# Physical Consistency Toolkit report","",
           "PCI is a relative post-hoc diagnostic; lower values indicate fewer adopted penalties.",
           "It does not establish compliance with governing physical laws or universal suitability.",
           "This run is not a reproduction of the published empirical results.","",
           f"Domain: {metadata.get('domain','imported surface masks')}. See run_metadata.json for axes and units.",
           "Roughness is an RMS first-derivative magnitude, not a curvature measure: steep smooth surfaces can score high.",
           "All models share one grid, mask, and maximum-roughness reference.","",
           "| Model | MVR | Rough norm | PBV | PCI | RMSE | Pareto |",
           "|---|---:|---:|---:|---:|---:|---|"]
    for r in table.to_dict("records"):
        lines.append(f"| {r['model_id']} | {r['mvr']:.6f} | {r['rough_norm']:.6f} | "
                     f"{r['pbv']:.6f} | {r['pci']:.6f} | {r.get('rmse','n/a')} | {r.get('pareto','n/a')} |")
    lines += ["","Results: metrics.csv, weight_sensitivity.csv, surfaces.csv, run_metadata.json.",
              "Holdout predictions/residuals, split assignments and rejection reports are local analysis outputs.",
              "Review them before sharing when using your own data.","",
              "Interpret rankings only for this cohort, domain, units, grid, criteria and normalization."]
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
