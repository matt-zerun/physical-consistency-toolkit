"""TOML input with strict keys and paths relative to the configuration file."""
import copy
from pathlib import Path
import tomllib
import numpy as np

DEFAULTS={
    "seed":42,"input":"synthetic.csv","output":"results","test_fraction":0.2,
    "split":"random","plots":True,"interactive":False,
    "validation":{"on_invalid":"error"},
    "eda":{"enabled":False,"moisture_threshold":None,"n_clusters":3,"contamination":0.02,"plots":True},
    "grid":{"shape":[41,41],"domain":"hull"},
    "pci":{"weights":[0.4,0.4,0.2],"bounds":[0.2,5.0],"directions":[1,1],"tolerance":0.0,
           "weight_variants":{"roughness_emphasis":[0.25,0.55,0.2]}},
    "models":[{"id":"SVR","family":"svr"},{"id":"RF","family":"rf"},
              {"id":"DT","family":"dt"},{"id":"KNN","family":"knn"}],
}
ALLOWED={"eda":{"enabled","moisture_threshold","n_clusters","contamination","plots"},
         "validation":{"on_invalid","ranges","min_moisture","material_group"},
         "grid":{"shape","domain","moisture_range","density_range"},
         "pci":{"weights","bounds","directions","tolerance","weight_variants"}}


def resolve_config(config):
    if not isinstance(config,dict) or set(config)-set(DEFAULTS)-{"column_map"}:
        raise ValueError("Unknown top-level configuration option")
    out=copy.deepcopy(DEFAULTS)
    for k,v in config.items():
        if k in ALLOWED:
            if not isinstance(v,dict) or set(v)-ALLOWED[k]:
                raise ValueError(f"Unknown option or invalid table: {k}")
            out[k].update(v)
        else:
            out[k]=copy.deepcopy(v)
    from .eda import _validate_options
    eda=out["eda"]
    if not isinstance(eda["enabled"],bool) or not isinstance(eda["plots"],bool):
        raise ValueError("eda.enabled and eda.plots must be boolean")
    _validate_options(eda["moisture_threshold"],eda["n_clusters"],eda["contamination"],out["seed"])
    if out["validation"].get("on_invalid") not in ("error","drop"):
        raise ValueError("validation.on_invalid must be error or drop")
    if out["split"] not in ("random","group"):
        raise ValueError("split must be random or group")
    if not np.isfinite(out["test_fraction"]) or not 0<out["test_fraction"]<1:
        raise ValueError("test_fraction must be between zero and one")
    if out["grid"]["domain"] not in ("hull","rectangle"):
        raise ValueError("grid.domain must be hull or rectangle")
    if not isinstance(out["seed"],int) or out["seed"]<0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(out["models"],list) or not out["models"]:
        raise ValueError("At least one model must be configured")
    for key in ["plots","interactive"]:
        if not isinstance(out[key],bool):
            raise ValueError(f"{key} must be true or false")
    return out


def read_config(path):
    p=Path(path).resolve()
    with p.open("rb") as f:
        return resolve_config(tomllib.load(f)),p.parent
