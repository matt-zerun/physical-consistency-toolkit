"""Run this file in Spyder after installing the package in its interpreter."""
from pathlib import Path
from physical_consistency import run_analysis

if __name__ == "__main__":
    # Resolves relative to this file, independently of Spyder's working directory.
    summary = run_analysis(Path(__file__).resolve().with_name("demo.toml"))
    print(summary)
