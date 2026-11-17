# Synthetic provenance

All bundled tables were created for this package independently of laboratory data.
They do not reproduce source record structures or empirical distributions. The
ranges and coefficients below were chosen pedagogically; they were not estimated
from the paper's records. Similarity to any measured value is incidental.

numpy.random.default_rng, seed 73021, n=240:

- w ~ Uniform(2,17), percent gravimetric water content.
- rho ~ Uniform(1.35,1.75), g/cm3 wet bulk density, independent of w.
- lambda_true = 0.35 + 0.9(1-exp(-w/8)) + 0.8(rho-1.3) + 0.025w(rho-1.3).
- lambda = lambda_true + Normal(0,0.045^2), W/(m K).

The function increases with both predictors, without claiming a constitutive soil
law. Identifier prefix SYN and is_synthetic=true mark every generated measurement.
Recreate with generate_synthetic().to_csv('synthetic.csv',index=False).

examples/synthetic_surfaces.csv contains six independently defined surfaces on
w=2..17 (21 nodes), rho=1.35..1.75 (19 nodes): constant, increasing affine plane,
decreasing affine plane, sinusoidal oscillations, checkerboard jumps and a constant
above the operational conductivity bound. Their exact definitions and generator
are in examples/generate_examples.py. They all share a full rectangle so they can
be compared jointly. All are synthetic; no fitted models or laboratory-derived
coefficients are used.

Tests generate further arrays analytically, including quadratic surfaces and
triangular hull masks. No test fixture contains observations from the study.

## Independent EDA demonstration data

`generate_synthetic_eda()` / `examples/synthetic_eda.csv` uses a **separate
pedagogical construction** and random seed 64021, with 180 invented moist states
(w uniformly 1.5–19.0%) and 45 invented below-threshold states
(w uniformly 0.05–0.85%). Wet bulk density is independently uniform 1.25–1.95
g/cm3. Illustrative conductivity is an arbitrary smooth function of moisture
and density with Gaussian noise for moist observations; dry examples use a
separate low-moisture illustrative formula. All coefficients are hard-coded,
not derived from empirical studies or laboratory distributions. This artificial
mixture demonstrates threshold diagnostics **without** implying that all
materials exhibit a corresponding natural dry/moist distinction.

Regenerate both example tables using `python examples/generate_examples.py` after
installing the package. Neither of these synthetic tables reproduces the
measurements or descriptive statistics in Energies 19(8), 1827.
