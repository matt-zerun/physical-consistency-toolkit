"""Run EDA on fictional records in Spyder after package installation."""
from pathlib import Path
from physical_consistency.eda import run_eda_file

HERE = Path(__file__).resolve().parent
summary = run_eda_file(HERE / "synthetic_eda.csv", HERE / "eda_spyder_results",
                       moisture_threshold=1.0, plots=True)
print(summary)
