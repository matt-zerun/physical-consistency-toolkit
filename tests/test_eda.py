"""Regression tests for public EDA, based only on analytical and synthetic inputs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from physical_consistency import run_analysis, run_eda_file
from physical_consistency.eda import (analyse_dataframe, descriptive_statistics,
                                      predictor_vif)
from physical_consistency.synthetic import generate_synthetic_eda
from physical_consistency.config import resolve_config


class EDATests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.base=Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()

    def test_reproducible_synthetic_and_no_original_data(self):
        a=generate_synthetic_eda(seed=51)
        pd.testing.assert_frame_equal(a,generate_synthetic_eda(seed=51))
        self.assertTrue(a.is_synthetic.all())
        self.assertEqual(set(a.columns),{"record_id","moisture_mass_pct", "bulk_density_g_cm3",
                                         "thermal_conductivity_w_mk","material_group","is_synthetic"})
        self.assertEqual((a.moisture_mass_pct<1.0).sum(),45)

    def test_descriptive_mathematics(self):
        df=pd.DataFrame({"record_id":["a","b","c","d"],
              "moisture_mass_pct":[1,2,3,4],"bulk_density_g_cm3":[1.3,1.5,1.7,1.9],
              "thermal_conductivity_w_mk":[2,3,4,5]})
        t=descriptive_statistics(df).set_index("variable")
        self.assertEqual(t.loc["moisture_mass_pct","n"],4)
        self.assertAlmostEqual(t.loc["moisture_mass_pct","mean"],2.5)
        self.assertAlmostEqual(t.loc["moisture_mass_pct","iqr"],1.5)
        self.assertAlmostEqual(t.loc["moisture_mass_pct","std_sample"],np.sqrt(5/3))
        self.assertAlmostEqual(t.loc["moisture_mass_pct","cv"],np.sqrt(5/3)/2.5)
        self.assertAlmostEqual(t.loc["moisture_mass_pct","skewness_adjusted"],0)
        self.assertAlmostEqual(t.loc["moisture_mass_pct","kurtosis_excess_adjusted"],-1.2)

    def test_vif_intercept_and_perfect_collinearity(self):
        x=np.arange(60)/10+1
        df=pd.DataFrame({"moisture_mass_pct":x,"bulk_density_g_cm3":1.3+0.1*x})
        t=predictor_vif(df)
        self.assertTrue((t.status=="unbounded_perfect_collinearity").all())
        self.assertTrue(t.vif.isna().all())
        rng=np.random.default_rng(1)
        df["bulk_density_g_cm3"]=rng.random(60)+1
        self.assertTrue((predictor_vif(df).vif>=1).all())

    def test_subsets_corr_pca_flags_no_outlier_removal(self):
        d=generate_synthetic_eda()
        out=self.base/"eda"
        result=analyse_dataframe(d,out,moisture_threshold=1,n_clusters=3,plots=False)
        self.assertEqual(result["sample_sizes"],{"all_valid":225,"below_threshold":45,"at_or_above_threshold":180})
        self.assertGreater(result["n_outliers_flagged"],0)
        self.assertLess(result["n_outliers_flagged"],180)
        self.assertFalse(result["outliers_removed"])
        self.assertFalse((out/"plots").exists())
        t=pd.read_csv(out/"descriptive_statistics.csv")
        self.assertEqual(len(t),9)
        self.assertEqual(t.loc[t.subset=="below_threshold","n"].iloc[0],45)
        self.assertEqual(len(pd.read_csv(out/"pca_scores.csv")),180)
        self.assertEqual(len(pd.read_csv(out/"outlier_flags.csv")),180)
        self.assertEqual(pd.read_csv(out/"pearson_correlation.csv").shape,(3,4))
        self.assertEqual(len(pd.read_csv(out/"pca_loadings.csv")),3)
        self.assertEqual(len(pd.read_csv(out/"pca_explained_variance.csv")),2)
        with self.assertRaises(ValueError):analyse_dataframe(d,out,plots=False)

    def test_outlier_flag_count_is_not_observation_count(self):
        d=generate_synthetic_eda(n_dry=12,n_moist=50)
        result=analyse_dataframe(d,self.base/"output",plots=False,contamination=.1)
        self.assertGreater(result["n_outliers_flagged"],0)
        self.assertLess(result["n_outliers_flagged"],62)
        self.assertEqual(result["n_outliers_flagged"],int(pd.read_csv(self.base/"output/outlier_flags.csv").is_outlier.sum()))
        self.assertEqual(len(pd.read_csv(self.base/"output/outlier_flags.csv")),62)

    def test_analytical_correlations(self):
        x=np.linspace(2,14,50)
        d=pd.DataFrame({"record_id":[str(i) for i in range(50)],"moisture_mass_pct":x,
             "bulk_density_g_cm3":np.sin(x)+1.7,"thermal_conductivity_w_mk":0.2*x})
        analyse_dataframe(d,self.base/"corr",plots=False)
        c=pd.read_csv(self.base/"corr/pearson_correlation.csv",index_col=0)
        self.assertAlmostEqual(c.loc["moisture_mass_pct","thermal_conductivity_w_mk"],1,places=10)

    def test_eda_with_invalid_rows_and_configured_column_map(self):
        d=generate_synthetic_eda(n_dry=10,n_moist=30).rename(columns={"moisture_mass_pct":"fictional_water"})
        d.loc[0,"fictional_water"]=-7
        f=self.base/"custom.csv";d.to_csv(f,index=False)
        map={"fictional_water":"moisture_mass_pct"}
        with self.assertRaisesRegex(ValueError,"rejected 1"):
            run_eda_file(f,self.base/"strict",column_map=map,plots=False)
        self.assertEqual(len(pd.read_csv(self.base/"strict/rejected_rows.csv")),1)
        result=run_eda_file(f,self.base/"drop",column_map=map,
                            validation={"on_invalid":"drop"},plots=False)
        self.assertEqual(result["sample_sizes"]["all_valid"],39)
        self.assertEqual(result["n_rejected"],1)

    def test_optional_plots_and_cli(self):
        d=generate_synthetic_eda(n_dry=15,n_moist=35)
        file=self.base/"sample.csv";d.to_csv(file,index=False)
        cmd=[sys.executable,"-m","physical_consistency","eda","--input",str(file),
             "--output",str(self.base/"cli"),"--moisture-threshold","1"]
        process=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
        self.assertEqual(process.returncode,0,process.stderr)
        for filename in ["pairplot.png","pearson_correlation.png","distributions.png",
                         "pca_clusters.png","below_threshold_distributions.png"]:
            self.assertTrue((self.base/"cli/plots"/filename).is_file(),filename)
        self.assertTrue(json.loads((self.base/"cli/eda_report.json").read_text())["source_sha256"])

    def test_eda_cli_config_paths(self):
        d=generate_synthetic_eda(n_dry=12,n_moist=32)
        d.to_csv(self.base/"sample.csv",index=False)
        (self.base/"eda.toml").write_text('input = "sample.csv"\noutput = "result"\n[eda]\nmoisture_threshold = 1.0\nplots = false\n',encoding="utf-8")
        env=dict(os.environ)
        env["PYTHONPATH"]=str(Path(__file__).resolve().parents[1]/"src")+os.pathsep+env.get("PYTHONPATH","")
        result=subprocess.run([sys.executable,"-m","physical_consistency","eda","--config",str(self.base/"eda.toml")],
                              cwd="/",env=env,capture_output=True,text=True,timeout=120)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue((self.base/"result/eda_report.json").is_file())
        self.assertFalse((self.base/"result/plots").exists())

    def test_eda_enabled_in_training_is_diagnostic_only(self):
        d=generate_synthetic_eda(n_dry=14,n_moist=56)
        d.to_csv(self.base/"training.csv",index=False)
        cfg={"input":"training.csv","output":"training_out","plots":False,
             "validation":{"min_moisture":1.0,"on_invalid":"drop"},
             "eda":{"enabled":True,"moisture_threshold":1.0}}
        t=run_analysis(cfg,base_dir=self.base)
        self.assertEqual(len(t),4)
        report=json.loads((self.base/"training_out/eda/eda_report.json").read_text())
        self.assertEqual(report["sample_sizes"],{"all_valid":70,"below_threshold":14,"at_or_above_threshold":56})
        meta=json.loads((self.base/"training_out/run_metadata.json").read_text())
        self.assertEqual(meta["accepted_rows"],56)
        self.assertTrue(meta["eda_enabled"])

    def test_unreasonable_config_rejected(self):
        with self.assertRaises(ValueError):resolve_config({"eda":{"n_clusters":1}})
        with self.assertRaises(ValueError):resolve_config({"eda":{"contamination":0.75}})
        with self.assertRaises(ValueError):resolve_config({"eda":{"moisture_threshold":-1}})
        with self.assertRaises(ValueError):resolve_config({"eda":{"new_option":True}})

    def test_no_pca_for_constant_variable(self):
        d=generate_synthetic_eda(n_dry=5,n_moist=12)
        d["thermal_conductivity_w_mk"]=2.0
        r=analyse_dataframe(d,self.base/"constant",plots=False)
        self.assertEqual(r["pca_components"],0)
        self.assertTrue(any("constant" in s for s in r["warnings"]))
