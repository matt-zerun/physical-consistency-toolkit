"""Physical Consistency Toolkit. All bundled examples are synthetic."""
__version__ = "0.2.0"
from .pci import components, compare_surfaces
from .surfaces import Surface, generate_grid, read_surfaces, write_surfaces
from .synthetic import generate_synthetic, generate_synthetic_eda
from .data import validate_data, load_data
from .eda import run_eda_file, analyse_dataframe, descriptive_statistics, predictor_vif
from .workflow import run_analysis
__all__ = ["components", "compare_surfaces", "Surface", "generate_grid", "read_surfaces",
           "write_surfaces", "generate_synthetic", "generate_synthetic_eda", "validate_data", "load_data", "run_analysis",
           "run_eda_file", "analyse_dataframe", "descriptive_statistics", "predictor_vif"]
