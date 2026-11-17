"""Terminal commands with the same public API used by Spyder scripts."""
import argparse
import json
from pathlib import Path
import sys
from .config import read_config
from .data import load_data,validate_data
from .synthetic import generate_synthetic
from .workflow import run_analysis,evaluate_surface_file
from .eda import run_eda_file


def main(argv=None):
    parser=argparse.ArgumentParser(description="Physical Consistency Toolkit (post-hoc PCI)")
    sub=parser.add_subparsers(dest="command",required=True)
    demo=sub.add_parser("demo",help="Run an entirely synthetic demonstration offline")
    demo.add_argument("--output",default="pci_demo")
    demo.add_argument("--no-plots",action="store_true")
    run=sub.add_parser("run",help="Analyse user data using a TOML configuration")
    run.add_argument("--config",required=True)
    validate=sub.add_parser("validate",help="Write row-level validation issues without fitting")
    validate.add_argument("--input",required=True)
    validate.add_argument("--report",default="validation_issues.csv")
    validate.add_argument("--config",help="Use this configuration's mapping and validation rules")
    eda=sub.add_parser("eda",help="Run publication-inspired exploratory diagnostics on public-schema data")
    eda.add_argument("--input",help="CSV or XLSX; mandatory unless supplied in --config")
    eda.add_argument("--output",help="New directory for EDA results; mandatory unless supplied in --config")
    eda.add_argument("--config",help="Optional TOML mapping, validation and EDA settings")
    eda.add_argument("--moisture-threshold",type=float,default=None,
                     help="Optional gravimetric percent threshold; below-threshold rows are descriptive only")
    eda.add_argument("--no-plots",action="store_true")
    surface=sub.add_parser("evaluate-surface",help="Evaluate one or more imported model surfaces jointly")
    surface.add_argument("--input",required=True)
    surface.add_argument("--output",required=True)
    surface.add_argument("--config",help="Use the PCI table and plot settings from this TOML file")
    surface.add_argument("--no-plots",action="store_true")
    args=parser.parse_args(argv)
    try:
        if args.command=="demo":
            base=Path(args.output).resolve()
            if base.exists() and any(base.iterdir()):
                raise ValueError("Demo output must be new or empty")
            base.mkdir(parents=True,exist_ok=True)
            generate_synthetic().to_csv(base/"synthetic.csv",index=False)
            result=run_analysis({"input":"synthetic.csv","output":"results","plots":not args.no_plots,
                                 "eda":{"enabled":True}},base_dir=base)
            print(result[["model_id","rmse","pci","pareto"]].to_string(index=False))
            print("Synthetic demonstration completed.")
        elif args.command=="run":
            result=run_analysis(args.config)
            print(result[["model_id","rmse","pci","pareto"]].to_string(index=False))
        elif args.command=="eda":
            cfg,root=read_config(args.config) if args.config else ({},Path.cwd())
            settings=cfg.get("eda",{})
            if args.input is None and args.config is None:
                raise ValueError("--input required when no --config is supplied")
            if args.output is None and args.config is None:
                raise ValueError("--output required when no --config is supplied")
            src=Path(args.input).resolve() if args.input else (root/cfg["input"]).resolve()
            out=Path(args.output).resolve() if args.output else (root/cfg["output"]).resolve()
            threshold=(args.moisture_threshold if args.moisture_threshold is not None
                       else settings.get("moisture_threshold"))
            result=run_eda_file(src,out,column_map=cfg.get("column_map"),
                        validation={k:v for k,v in cfg.get("validation",{}).items() if k!="min_moisture"},
                        moisture_threshold=threshold,
                        n_clusters=settings.get("n_clusters",3),
                        contamination=settings.get("contamination",0.02),
                        seed=cfg.get("seed",42),plots=not args.no_plots and settings.get("plots",True))
            print(json.dumps({"EDA completed": True,"sample_sizes":result["sample_sizes"],
                              "n_outliers_flagged":result["n_outliers_flagged"]}))
        elif args.command=="validate":
            cfg,_=read_config(args.config) if args.config else ({},None)
            settings={**cfg.get("validation",{}),"on_invalid":"drop"}
            frame=load_data(args.input,cfg.get("column_map"))
            accepted,rejected=validate_data(frame,**settings)
            target=Path(args.report)
            if target.exists():
                raise ValueError("Validation report already exists; choose another filename")
            target.parent.mkdir(parents=True,exist_ok=True)
            rejected.to_csv(target,index=False)
            print(json.dumps({"accepted":len(accepted),"rejected":len(rejected)}))
            return 1 if len(rejected) else 0
        else:
            cfg,_=read_config(args.config) if args.config else ({},None)
            result=evaluate_surface_file(args.input,args.output,pci=cfg.get("pci"),
                        plots=not args.no_plots and cfg.get("plots",True),interactive=cfg.get("interactive",False))
            print(result.to_string(index=False))
    except (ValueError,KeyError,TypeError,OSError,ImportError) as exc:
        print(f"Error: {exc}",file=sys.stderr)
        return 2
    return 0
