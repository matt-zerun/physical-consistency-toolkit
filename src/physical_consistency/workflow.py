"""Orchestrate validation, leakage-aware fitting, surfaces and post-hoc PCI."""
from pathlib import Path
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GroupShuffleSplit
from .config import resolve_config,read_config
from .data import FEATURES,TARGET,load_data,validate_data
from .models import fit_model
from .surfaces import Surface,generate_grid,write_surfaces,read_surfaces,safe_name
from .pci import compare_surfaces,gradients
from .evaluation import accuracy,pareto_mask
from .reporting import environment,save_json,file_hash,write_report


def _new_output(path):
    p=Path(path).resolve()
    if p.exists() and any(p.iterdir()):
        raise ValueError("Output directory must be new or empty; choose another output path")
    p.mkdir(parents=True,exist_ok=True)
    return p


def _pci_tables(surfaces,criteria):
    settings=copy.deepcopy(criteria)
    variants=settings.pop("weight_variants",{})
    table=compare_surfaces(surfaces,**settings)
    parts=[table.assign(weight_variant="baseline")]
    for name,weights in variants.items():
        safe_name(name)
        parts.append(compare_surfaces(surfaces,**{**settings,"weights":weights}).assign(weight_variant=name))
    return table,pd.concat(parts,ignore_index=True)


def _surface_diagnostics(surfaces,output,bounds,directions,tolerance):
    rows=[]
    for name,s in surfaces.items():
        dw,dd,valid=gradients(s)
        vals=s.z.ravel()
        rows.append(pd.DataFrame({"model_id":name,FEATURES[0]:s.points[:,0],FEATURES[1]:s.points[:,1],
            "inside_domain":s.mask.ravel(),"gradient_eligible":valid.ravel(),
            "derivative_moisture":dw.ravel(),"derivative_density":dd.ravel(),
            "monotonicity_violation_moisture":((dw*directions[0]<-tolerance)&valid).ravel(),
            "monotonicity_violation_density":((dd*directions[1]<-tolerance)&valid).ravel(),
            "bounds_violation":((vals<bounds[0])|(vals>bounds[1]))&s.mask.ravel()}))
    pd.concat(rows,ignore_index=True).to_csv(output/"surface_diagnostics.csv",index=False)


def run_analysis(config, *, base_dir=None):
    """Run from a TOML path or config dict; return the summary DataFrame.

    Config-relative paths work in terminal and Spyder. For dicts, base_dir is
    explicit or defaults to the current directory. No model pickle is written.
    """
    if isinstance(config,(str,Path)):
        cfg,base=read_config(config)
    else:
        cfg=resolve_config(config);base=Path(base_dir or Path.cwd()).resolve()
    source=(base/cfg["input"]).resolve()
    raw=load_data(source,cfg.get("column_map"))
    # Validate with a full report first, then enforce the requested failure policy.
    validation={**cfg["validation"],"on_invalid":"drop"}
    clean,rejected=validate_data(raw,**validation)
    output=_new_output(base/cfg["output"])
    rejected.to_csv(output/"rejected_rows.csv",index=False)
    save_json(output/"config.json",cfg)
    if len(rejected) and cfg["validation"].get("on_invalid","error")=="error":
        raise ValueError("Invalid rows found; rejected_rows.csv written. Review data or select on_invalid='drop'")
    if len(clean)<12:
        raise ValueError("At least 12 valid records are required")
    if cfg["eda"]["enabled"]:
        # Pre-modelling diagnostics on valid material-group records; recover
        # below-threshold rows for descriptive EDA without passing them to ML.
        eda_rules={**cfg["validation"],"on_invalid":"drop"}
        eda_rules.pop("min_moisture",None)
        eda_clean,eda_issues=validate_data(raw,**eda_rules)
        from .eda import analyse_dataframe
        ecfg=cfg["eda"]
        analyse_dataframe(eda_clean,output/"eda",
                          moisture_threshold=ecfg["moisture_threshold"],
                          n_clusters=ecfg["n_clusters"],
                          contamination=ecfg["contamination"],seed=cfg["seed"],
                          plots=ecfg["plots"] and cfg["plots"],
                          rejected_rows=eda_issues,source_sha256=file_hash(source))
    if "specimen_id" in clean and clean.specimen_id.duplicated().any() and cfg["split"]!="group":
        raise ValueError("Repeated specimen_id requires split='group' to prevent leakage")
    X=clean[FEATURES].to_numpy();y=clean[TARGET].to_numpy()
    idx=np.arange(len(clean))
    groups=None
    if cfg["split"]=="group":
        if "specimen_id" not in clean or clean.specimen_id.nunique()<3:
            raise ValueError("Group split requires specimen_id and >=3 distinct groups")
        groups=clean.specimen_id.astype(str).to_numpy()
        train,test=next(GroupShuffleSplit(n_splits=1,test_size=cfg["test_fraction"],random_state=cfg["seed"]).split(X,y,groups))
    else:
        train,test=train_test_split(idx,test_size=cfg["test_fraction"],random_state=cfg["seed"])
    if min(len(train),len(test))<2:
        raise ValueError("Training and test subsets each require >=2 rows")
    split=clean[["record_id"]].copy()
    split["subset"]="train";split.loc[split.index[test],"subset"]="test"
    if groups is not None:
        split["specimen_id"]=groups
    split.to_csv(output/"split_assignments.csv",index=False)
    grid=cfg["grid"]
    ranges=[grid.get("moisture_range",[float(X[train,0].min()),float(X[train,0].max())]),
            grid.get("density_range",[float(X[train,1].min()),float(X[train,1].max())])]
    template=generate_grid(*ranges,shape=grid["shape"],support=X[train] if grid["domain"]=="hull" else None)
    surfaces={};acc=[];predictions=[];fitted={}
    for spec in cfg["models"]:
        name=safe_name(spec["id"])
        if name.casefold() in {s.casefold() for s in surfaces}:
            raise ValueError("Model IDs must be unique, including on Windows")
        model,info=fit_model(spec,X[train],y[train],seed=cfg["seed"],groups=groups[train] if groups is not None else None)
        pred=model.predict(X[test])
        acc.append({"model_id":name,**accuracy(y[test],pred)})
        surfaces[name]=Surface(template.moisture,template.density,
                               model.predict(template.points).reshape(template.z.shape),template.mask)
        predictions.append(pd.DataFrame({"model_id":name,"record_id":clean.iloc[test].record_id.to_numpy(),
            "observed_w_mk":y[test],"predicted_w_mk":pred,"residual_w_mk":pred-y[test]}))
        fitted[name]=info
    table,sensitivity=_pci_tables(surfaces,cfg["pci"])
    table=table.merge(pd.DataFrame(acc),on="model_id",validate="one_to_one")
    table["pareto"]=pareto_mask(table[["rmse","pci"]])
    sensitivity=sensitivity.merge(table[["model_id","rmse"]],on="model_id",validate="many_to_one")
    sensitivity["pareto"]=False
    for _,g in sensitivity.groupby("weight_variant"):
        sensitivity.loc[g.index,"pareto"]=pareto_mask(g[["rmse","pci"]])
    predictions=pd.concat(predictions,ignore_index=True)
    metadata={"software":"physical-consistency-toolkit 0.2.0","environment":environment(),
        "input_sha256":file_hash(source),"accepted_rows":len(clean),"rejected_rows":len(rejected),
        "n_train":len(train),"n_test":len(test),"domain":grid["domain"],"domain_source":"training predictors only",
        "axis_units":["percent gravimetric water content","g/cm3 wet bulk density"],"target_unit":"W/(m K)",
        "grid_shape":list(template.z.shape),"moisture_axis":template.moisture.tolist(),"density_axis":template.density.tolist(),
        "rough_ref":float(table.rough_ref.iloc[0]),"comparison_cohort":list(surfaces),
        "criteria":cfg["pci"],"stencil":"forward, backward at unavailable forward neighbour",
        "fitted_models":fitted,"synthetic_status":"marked_all_synthetic" if "is_synthetic" in clean and clean.is_synthetic.astype(str).str.lower().eq("true").all() else "user_supplied_unverified",
        "publication_results_reproduced":False,"eda_enabled":cfg["eda"]["enabled"]}
    table.to_csv(output/"metrics.csv",index=False)
    sensitivity.to_csv(output/"weight_sensitivity.csv",index=False)
    predictions.to_csv(output/"holdout_predictions.csv",index=False)
    write_surfaces(surfaces,output/"surfaces.csv")
    _surface_diagnostics(surfaces,output,cfg["pci"]["bounds"],cfg["pci"]["directions"],cfg["pci"]["tolerance"])
    save_json(output/"run_metadata.json",metadata)
    write_report(output,table,metadata)
    if cfg["plots"]:
        from .visualization import plot_results
        plot_results(surfaces,table,predictions,output/"plots",interactive=cfg["interactive"])
    return table


def evaluate_surface_file(input_path, output_path, *, pci=None, plots=True, interactive=False):
    """Evaluate external models jointly; no observations, training or accuracy claim."""
    surfaces=read_surfaces(input_path)
    criteria=resolve_config({"pci":pci or {}})["pci"]
    table,sensitivity=_pci_tables(surfaces,criteria)
    output=_new_output(output_path)
    table.to_csv(output/"metrics.csv",index=False)
    sensitivity.to_csv(output/"weight_sensitivity.csv",index=False)
    write_surfaces(surfaces,output/"surfaces.csv")
    _surface_diagnostics(surfaces,output,criteria["bounds"],criteria["directions"],criteria["tolerance"])
    first=next(iter(surfaces.values()))
    metadata={"environment":environment(),"input_sha256":file_hash(input_path),"criteria":criteria,
              "comparison_cohort":list(surfaces),"rough_ref":float(table.rough_ref.iloc[0]),
              "moisture_axis":first.moisture.tolist(),"density_axis":first.density.tolist(),
              "axis_units":["percent gravimetric water content","g/cm3 wet bulk density"],"target_unit":"W/(m K)",
              "publication_results_reproduced":False,"accuracy":"not available without observations",
              "stencil":"forward, backward at unavailable forward neighbour"}
    save_json(output/"run_metadata.json",metadata)
    write_report(output,table,metadata)
    if plots:
        from .visualization import plot_results
        plot_results(surfaces,table,None,output/"plots",interactive=interactive)
    return table
