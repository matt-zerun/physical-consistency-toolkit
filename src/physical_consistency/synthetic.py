"""Independent illustrative generator; no empirical calibration or source records."""
import numpy as np
import pandas as pd


def generate_synthetic(n=240, seed=73021):
    """Generate synthetic gravimetric water content, wet bulk density and lambda.

    Deliberately selected pedagogical ranges: w=2..17 percent, rho=1.35..1.75
    g/cm3. Noise is N(0, 0.045**2) W/(m K). See docs/synthetic.md.
    """
    if not isinstance(n, int) or n < 12:
        raise ValueError("n must be an integer >= 12")
    rng = np.random.default_rng(seed)
    w = rng.uniform(2, 17, n)
    d = rng.uniform(1.35, 1.75, n)
    truth = 0.35 + 0.9*(1-np.exp(-w/8)) + 0.8*(d-1.3) + 0.025*w*(d-1.3)
    return pd.DataFrame({
        "record_id": [f"SYN-{i:05d}" for i in range(n)],
        "moisture_mass_pct": w,
        "bulk_density_g_cm3": d,
        "thermal_conductivity_w_mk": truth + rng.normal(0, 0.045, n),
        "material_group": "synthetic_sand", "is_synthetic": True,
    })


def generate_synthetic_eda(n_moist=180, n_dry=45, seed=64021):
    """Create entirely synthetic dry/moist illustrations for the EDA CLI.

    Values are specified for demonstration, never fitted to observations or to
    reported empirical distributions. The below-threshold class is an invented
    teaching example, not evidence about physical drying mechanisms.
    """
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 3 for n in [n_moist,n_dry]):
        raise ValueError("n_moist and n_dry must be integers >=3")
    rng = np.random.default_rng(seed)
    wm = rng.uniform(1.5, 19.0, n_moist)
    wd = rng.uniform(0.05, 0.85, n_dry)
    dm = rng.uniform(1.25, 1.95, n_moist)
    dd = rng.uniform(1.25, 1.95, n_dry)
    ym = 0.48+0.85*(1-np.exp(-wm/7.5))+0.95*(dm-1.2)+rng.normal(0,0.055,n_moist)
    yd = 0.22+0.22*(dd-1.2)+0.09*wd+rng.normal(0,0.035,n_dry)
    w = np.r_[wd,wm]; density=np.r_[dd,dm]; lam=np.r_[yd,ym]
    idx=rng.permutation(len(w))
    return pd.DataFrame({"record_id":[f"SYN-EDA-{i:05d}" for i in range(len(w))],
        "moisture_mass_pct":w[idx],"bulk_density_g_cm3":density[idx],
        "thermal_conductivity_w_mk":lam[idx],
        "material_group":"synthetic_demo_material","is_synthetic":True})
