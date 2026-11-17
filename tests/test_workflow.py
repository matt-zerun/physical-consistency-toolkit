"""Technical and integration tests; uses only newly generated synthetic data."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from physical_consistency import (generate_synthetic,validate_data,load_data,Surface,
    generate_grid,read_surfaces,write_surfaces,run_analysis)
from physical_consistency.evaluation import accuracy,pareto_mask
from physical_consistency.models import make_model,fit_model
from physical_consistency.config import resolve_config
from physical_consistency.workflow import evaluate_surface_file


class TechnicalTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_import_and_version(self):
        from physical_consistency import __version__
        self.assertEqual(__version__,"0.2.0")
    def test_generator_reproducible_and_no_global_rng(self):
        np.random.seed(123);a=np.random.rand()
        f=generate_synthetic(40);g=generate_synthetic(40)
        pd.testing.assert_frame_equal(f,g)
        self.assertFalse(f.equals(generate_synthetic(40,seed=2)))
        np.random.seed(123);self.assertEqual(a,np.random.rand())
        self.assertTrue(f.is_synthetic.all())
    def test_csv_and_column_map(self):
        f=generate_synthetic(40);f=f.rename(columns={"moisture_mass_pct":"my_water"})
        p=self.base/"x.csv";f.to_csv(p,index=False)
        clean,issues=validate_data(load_data(p,{"my_water":"moisture_mass_pct"}))
        self.assertEqual(len(clean),40);self.assertTrue(issues.empty)
    def test_xlsx_optional(self):
        if not importlib.util.find_spec("openpyxl"):self.skipTest("openpyxl unavailable")
        f=generate_synthetic(40);p=self.base/"x.xlsx";f.to_excel(p,index=False)
        self.assertEqual(len(validate_data(load_data(p))[0]),40)
    def test_missing_column_and_schema(self):
        with self.assertRaises(ValueError):validate_data(pd.DataFrame({"other":[1]}))
    def test_invalid_and_duplicate_rows_reported(self):
        f=generate_synthetic(40);f.loc[0,"bulk_density_g_cm3"]=np.nan
        f.loc[1,"record_id"]=f.loc[2,"record_id"]
        f.loc[3,"moisture_mass_pct"]=-1
        with self.assertRaises(ValueError):validate_data(f)
        clean,issues=validate_data(f,on_invalid="drop")
        self.assertEqual(len(clean),36);self.assertEqual(len(issues),4)
        self.assertIn("duplicate_record_id",";".join(issues.reasons))
    def test_duplicate_measurement_different_ids(self):
        f=generate_synthetic(40);cols=["moisture_mass_pct","bulk_density_g_cm3","thermal_conductivity_w_mk"]
        f.loc[1,cols]=f.loc[0,cols].to_numpy()
        self.assertEqual(len(validate_data(f,on_invalid="drop")[1]),2)
    def test_material_selection_not_implicit_pooling(self):
        f=generate_synthetic(40);f.loc[0,"material_group"]="other"
        with self.assertRaises(ValueError):validate_data(f)
        clean,issues=validate_data(f,on_invalid="drop",material_group="synthetic_sand")
        self.assertEqual(len(issues),1);self.assertEqual(len(clean),39)
    def test_grid_orientation_and_hull(self):
        s=generate_grid([0,1],[1,2],shape=[5,7],support=np.array([[0,1],[1,1],[0,2]]))
        self.assertEqual(s.z.shape,(5,7))
        self.assertTrue(np.array_equal(s.points[:7,0],np.zeros(7)))
        self.assertFalse(s.mask[-1,-1]);self.assertTrue(s.mask[0,0])
    def test_collinear_hull_rejected(self):
        with self.assertRaises(ValueError):generate_grid([0,1],[0,1],support=[[0,0],[0.5,0.5],[1,1]])
    def test_surface_csv_roundtrip(self):
        s=generate_grid([0,1],[1,2],shape=[5,7]);s=Surface(s.moisture,s.density,s.z+1)
        p=self.base/"surfaces.csv";write_surfaces({"model":s},p)
        actual=read_surfaces(p)["model"]
        np.testing.assert_allclose(actual.z,s.z)
    def test_surface_duplicate_and_incomplete_rejected(self):
        s=generate_grid([0,1],[1,2],shape=[5,7]);p=self.base/"s.csv";write_surfaces({"m":s},p)
        f=pd.read_csv(p);pd.concat([f,f.iloc[:1]]).to_csv(p,index=False)
        with self.assertRaises(ValueError):read_surfaces(p)
        f.iloc[:-1].to_csv(p,index=False)
        with self.assertRaises(ValueError):read_surfaces(p)
    def test_fit_all_required_models(self):
        f=generate_synthetic(40);X=f[["moisture_mass_pct","bulk_density_g_cm3"]].to_numpy()
        y=f.thermal_conductivity_w_mk.to_numpy()
        for family in ["svr","rf","dt","knn"]:
            m=make_model(family);m.fit(X[:30],y[:30])
            self.assertTrue(np.isfinite(m.predict(X[30:])).all())
    def test_scaling_fits_training_only(self):
        X=np.array([[0,1],[2,1],[4,2],[6,2]],float);y=np.arange(4.)
        m=make_model("svr");m.fit(X,y)
        np.testing.assert_allclose(m.steps[0][1].mean_,X.mean(axis=0))
        m.predict([[1000,2000]])
        np.testing.assert_allclose(m.steps[0][1].mean_,X.mean(axis=0))
    def test_training_cv(self):
        f=generate_synthetic(60);X=f[["moisture_mass_pct","bulk_density_g_cm3"]].to_numpy();y=f.thermal_conductivity_w_mk.to_numpy()
        m,info=fit_model({"id":"tuned","family":"svr","cv_folds":3,"search_grid":{"C":[1,10]}},X,y)
        self.assertTrue(np.isfinite(info["cv_best_r2"]))
    def test_accuracy_and_pareto(self):
        r=accuracy([1,2,3],[1,2,4]);self.assertAlmostEqual(r["mse"],1/3)
        self.assertIsNone(accuracy([0,1],[0,2])["mape_pct"])
        self.assertIsNone(accuracy([1,1],[1,1])["r2"])
        np.testing.assert_array_equal(pareto_mask([[1,2],[2,1],[3,3],[1,2]]),[1,1,0,1])
    def test_full_api_and_output(self):
        generate_synthetic(80).to_csv(self.base/"data.csv",index=False)
        t=run_analysis({"input":"data.csv","output":"out","plots":False},base_dir=self.base)
        self.assertEqual(len(t),4)
        for p in ["metrics.csv","surfaces.csv","holdout_predictions.csv","split_assignments.csv",
                  "surface_diagnostics.csv","weight_sensitivity.csv","config.json","run_metadata.json","report.md","rejected_rows.csv"]:
            self.assertTrue((self.base/"out"/p).is_file(),p)
        meta=json.loads((self.base/"out/run_metadata.json").read_text())
        self.assertEqual(meta["n_test"],16)
        self.assertFalse(meta["publication_results_reproduced"])
        with self.assertRaises(ValueError):run_analysis({"input":"data.csv","output":"out"},base_dir=self.base)
    def test_invalid_run_writes_report_then_stops(self):
        f=generate_synthetic(40);f.loc[0,"bulk_density_g_cm3"]=-1;f.to_csv(self.base/"d.csv",index=False)
        with self.assertRaises(ValueError):run_analysis({"input":"d.csv","output":"bad"},base_dir=self.base)
        self.assertEqual(len(pd.read_csv(self.base/"bad/rejected_rows.csv")),1)
    def test_grouped_split_and_cv(self):
        f=generate_synthetic(80);f["specimen_id"]=[f"S{i//4}" for i in range(80)]
        f.to_csv(self.base/"group.csv",index=False)
        cfg={"input":"group.csv","output":"random","plots":False}
        with self.assertRaises(ValueError):run_analysis(cfg,base_dir=self.base)
        cfg.update(split="group",output="group",models=[{"id":"svr","family":"svr","cv_folds":3,"search_grid":{"C":[1,10]}}])
        run_analysis(cfg,base_dir=self.base)
        split=pd.read_csv(self.base/"group/split_assignments.csv")
        a=set(split[split.subset=="train"].specimen_id);b=set(split[split.subset=="test"].specimen_id)
        self.assertFalse(a&b)
    def test_cli_demo_offline(self):
        # Network connections are forbidden by a child-process sitecustomize hook.
        hook=self.base/"hook";hook.mkdir()
        (hook/"sitecustomize.py").write_text("import socket\ndef denied(*a, **kw): raise RuntimeError('network forbidden')\nsocket.socket.connect=denied\nsocket.create_connection=denied\n")
        env=dict(os.environ);env["PYTHONPATH"]=str(hook)+os.pathsep+env.get("PYTHONPATH","")
        p=subprocess.run([sys.executable,"-m","physical_consistency","demo","--output",str(self.base/"offline")],
                          env=env,capture_output=True,text=True,timeout=180)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertTrue((self.base/"offline/results/plots/rmse_pci.png").is_file())
    def test_evaluate_surface_mode(self):
        s=generate_grid([0,1],[1,2],shape=[5,7]);s=Surface(s.moisture,s.density,s.z+1)
        p=self.base/"s.csv";write_surfaces({"a":s},p)
        t=evaluate_surface_file(p,self.base/"eval",plots=False)
        self.assertEqual(t.pci.iloc[0],0)
    def test_mapped_text_identifiers_preserve_leading_zeros(self):
        f=generate_synthetic(40).rename(columns={"record_id":"my_key"})
        f["my_key"]=[f"{i:04d}" for i in range(40)]
        p=self.base/"mapped.csv";f.to_csv(p,index=False)
        self.assertEqual(load_data(p,{"my_key":"record_id"}).record_id.iloc[0],"0000")
    def test_invalid_rejection_policy_fails(self):
        with self.assertRaises(ValueError):resolve_config({"validation":{"on_invalid":"typo"}})
    def test_config_unknown_keys_fail(self):
        with self.assertRaises(ValueError):resolve_config({"typo":True})
        with self.assertRaises(ValueError):resolve_config({"grid":{"typo":True}})
    def test_toml_paths_independent_of_working_directory(self):
        generate_synthetic(40).to_csv(self.base/"data.csv",index=False)
        config=self.base/"test.toml"
        config.write_text('input="data.csv"\noutput="toml_results"\nplots=false\n[[models]]\nid="tree"\nfamily="dt"\n')
        original=Path.cwd()
        try:
            os.chdir(self.base.parent)
            result=run_analysis(config)
        finally:
            os.chdir(original)
        self.assertEqual(len(result),1)
        self.assertTrue((self.base/"toml_results/metrics.csv").exists())
    def test_optional_xgboost(self):
        if not importlib.util.find_spec("xgboost"):self.skipTest("optional xgboost not installed")
        f=generate_synthetic(40);m=make_model("xgb",{"n_estimators":5})
        X=f[["moisture_mass_pct","bulk_density_g_cm3"]].to_numpy();y=f.thermal_conductivity_w_mk.to_numpy()
        m.fit(X,y);self.assertEqual(m.predict(X).shape,(40,))
    def test_optional_xgboost_full_export(self):
        if not importlib.util.find_spec("xgboost"):self.skipTest("optional xgboost not installed")
        generate_synthetic(40).to_csv(self.base/"xgb.csv",index=False)
        result=run_analysis({"input":"xgb.csv","output":"xgb_out","plots":False,
            "models":[{"id":"XGB","family":"xgb","params":{"n_estimators":5}}]},base_dir=self.base)
        self.assertEqual(len(result),1)
        metadata=json.loads((self.base/"xgb_out/run_metadata.json").read_text())
        self.assertEqual(metadata["fitted_models"]["XGB"]["fitted_estimator_params"]["missing"],"NaN")
    def test_optional_interactive(self):
        if not importlib.util.find_spec("plotly"):self.skipTest("optional plotly not installed")
        s=generate_grid([0,1],[1,2],shape=[5,7]);p=self.base/"s.csv";write_surfaces({"m":s},p)
        evaluate_surface_file(p,self.base/"interactive",interactive=True)
        html=(self.base/"interactive/plots/m_3d.html").read_text()
        self.assertIn("plotly.js",html.lower())

if __name__=="__main__":unittest.main()
