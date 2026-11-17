# PCI mathematics and numerical contract

## Authority and scope

The supplied published article, Section 2.6 equation (1), and supplied supplement,
Section S2 equations S1-S4, are the methodological authority. The package implements
these equations. It does not claim equality to any historical executable or
published numeric result. No real-data comparison has been made.

Let Z[i,j] be predicted thermal conductivity at gravimetric moisture w[i] (percent)
and wet bulk density r[j] (g/cm3), on strictly increasing regular axes. An indicator
I(condition) is 1 when true, otherwise 0.

- MVR = (mean(I(dw < 0)) + mean(I(dr < 0))) / 2.
- Rough = sqrt(mean(dw^2) + mean(dr^2)).
- Rough_ref = maximum Rough in the **entire comparison cohort**.
- Rough_norm = Rough / Rough_ref.
- PBV = mean(I(Z < lambda_min or Z > lambda_max)).
- PCI = alpha MVR + beta Rough_norm + gamma PBV.

Defaults: alpha=0.4, beta=0.4, gamma=0.2; lambda_min=0.2 and lambda_max=5.0 W/(m K).
Weights must be finite, nonnegative and sum to one; endpoints are admissible.
These are operational diagnostic bounds for the original context, not universal
thermophysical limits or data cleaning thresholds.

## Derivatives and domain boundaries

The supplement says first-order finite differences on a regular evaluation grid.
It does not uniquely specify node alignment, edge treatment or exact resolution.
Here dw[i,j]=(Z[i+1,j]-Z[i,j])/(w[i+1]-w[i]) wherever the forward neighbour belongs
to the domain; otherwise an immediate backward in-domain neighbour is used.
The last rectangular row uses a backward difference. Density is handled analogously.
Thus the final edge difference is represented at both adjacent nodes. This is a
node-based average over all rectangular grid points, consistent with S2's stated
averaging convention, with an explicitly chosen first-order stencil.

No derivative uses flattened surface ordering, unit grid spacing, rescaled axes,
Gaussian smoothing or curvature/Laplacian penalties. This is not numpy.gradient's
default centred interior stencil. The first-order stencil is an implementation
choice; historical equality is NOT VERIFIED.

The default hull is built from training predictors only. At a hull boundary a
missing forward neighbour causes a backward fallback if that neighbour is inside.
A node with no immediate in-domain neighbour on either side of an axis is excluded
from both derivative averages. Both gradient terms and both monotonicity fractions
use the same common eligible node set. PBV uses all included domain nodes. Counts
are exported. This masked-domain extension is not specified in S2; inspect counts
and refine the grid if too few nodes support gradients. In a full rectangle every
node is eligible and every expectation uses all grid nodes.

## Numerical and interpretative limitations

A constant-only cohort has Rough_ref=0; this package defines Rough_norm=0. S2 does
not define that degenerate case. In a single nonconstant surface cohort Rough_norm
is always 1; comparative PCI requires a meaningful set of surfaces evaluated
jointly. Changing the cohort changes all normalised roughness scores.

The roughness formula combines derivatives in two different predictor units.
It is coordinate/unit-dependent and is not a dimensionless physical invariant.
Keep percent and g/cm3 conventions fixed. Even smooth steep planes are penalised;
the metric is not a pure measure of oscillation. Differencing and grid resolution
can alter the index for discontinuous regressors. Repeat at several resolutions
and report sensitivity rather than claiming numerical convergence for tree jumps.

Hull membership is a geometric interpolation screen; it does not identify holes,
poorly sampled regions, impossible saturation states or reliable uncertainty.
Rectangle mode can include unobserved predictor combinations and extrapolation.
The package does not impose a saturation relation or validate a soil's constitutive
physics; the user must choose the applicable domain. Holdout RMSE is computed on
all held-out observations; it is not silently restricted to the training hull.

## Configurable extensions

Directions +1, -1 and 0 mean nondecreasing, nonincreasing and unconstrained.
If one axis is unconstrained, MVR averages the active axis only; with no active
axes MVR=0. Both derivative terms still enter Rough. A nonzero tolerance changes
the strict published test; state it explicitly. Other materials require justified
bounds, directions and domains. These choices do not automatically generalise the
original assumptions about quartz-rich sands.

The provided roughness_emphasis variant [0.25,0.55,0.2] is a new illustrative
sensitivity choice, not a reconstruction of Figure S2's alternative coefficients.
S2.2 does not specify their numerical values. An old executable's [0.4,0.8,0.2]
sums to 1.4 and is rejected here. No silent normalisation is applied.

PCI stays entirely post hoc. Inner CV optimises R2 on training rows; independent
holdout metrics are computed once. Pareto comparisons minimise RMSE and PCI; ties
are retained. They are diagnostic comparisons, not automatic model endorsements.
