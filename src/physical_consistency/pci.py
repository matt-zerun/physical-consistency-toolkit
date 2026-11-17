"""Supplement S2 equations S1-S4 with explicit first-order stencil choices."""
import numpy as np
import pandas as pd
from .surfaces import Surface

DEFAULT_WEIGHTS=(0.4,0.4,0.2)


def gradients(surface):
    """Forward coordinate-aware differences; backward at unavailable forward edges.

    At a hull boundary only in-domain immediate neighbours may contribute.
    A node without a neighbour in either direction is ineligible on that axis.
    Returns dw, dd and their common eligibility mask. No flattening or smoothing.
    """
    result=[]
    for axis,coords in enumerate([surface.moisture,surface.density]):
        z=np.moveaxis(surface.z,axis,0)
        m=np.moveaxis(surface.mask,axis,0)
        delta=np.diff(z,axis=0)/np.diff(coords)[:,None]
        edges=m[1:] & m[:-1]
        delta=np.where(edges,delta,np.nan)
        g=np.full(z.shape,np.nan)
        g[:-1]=delta
        fallback=np.full(z.shape,np.nan); fallback[1:]=delta
        g=np.where(np.isfinite(g),g,fallback)
        result.append(np.moveaxis(g,0,axis))
    common=surface.mask & np.isfinite(result[0]) & np.isfinite(result[1])
    if not common.any():
        raise ValueError("No nodes support derivatives on both axes; enlarge/refine the domain")
    return result[0],result[1],common


def components(surface, *, bounds=(0.2,5.0), directions=(1,1), tolerance=0.0):
    """Compute raw MVR, gradient RMS and PBV, plus eligible node counts.

    directions: +1 nondecreasing, -1 nonincreasing, 0 unconstrained per axis.
    All two gradient terms still enter roughness even if monotonicity is disabled.
    Bounds endpoints are admissible. Default tolerance is the published strict zero.
    """
    if len(bounds)!=2 or not np.isfinite(bounds).all() or bounds[0]>=bounds[1]:
        raise ValueError("bounds must contain two finite increasing values")
    if len(directions)!=2 or any(x not in (-1,0,1) for x in directions):
        raise ValueError("directions must contain two values from -1,0,1")
    if not np.isfinite(tolerance) or tolerance<0:
        raise ValueError("tolerance must be finite and nonnegative")
    dw,dd,valid=gradients(surface)
    violations=[float(np.mean(g[valid]*sign < -tolerance)) if sign else 0.0
                for g,sign in zip([dw,dd],directions)]
    active=sum(sign!=0 for sign in directions)
    mvr=sum(violations)/active if active else 0.0
    rough=float(np.sqrt(np.mean(dw[valid]**2)+np.mean(dd[valid]**2)))
    vals=surface.z[surface.mask]
    return {"mvr":mvr,"mvr_moisture":violations[0],"mvr_density":violations[1],
            "roughness":rough,"pbv":float(np.mean((vals<bounds[0])|(vals>bounds[1]))),
            "domain_nodes":int(surface.mask.sum()),"gradient_nodes":int(valid.sum()),
            "total_grid_nodes":int(surface.z.size)}


def compare_surfaces(surfaces, *, weights=DEFAULT_WEIGHTS, **criteria):
    """Compute PCI jointly with the maximum roughness of this comparison cohort.

    All surfaces must use the identical lattice and mask. A constant-only cohort
    has Rough_ref=0 and Rough_norm=0 (documented zero-denominator extension).
    One surface can be evaluated, but its nonzero Rough_norm is necessarily 1.
    """
    a=np.asarray(weights,dtype=float)
    if a.shape!=(3,) or not np.isfinite(a).all() or np.any(a<0) or not np.isclose(a.sum(),1,rtol=0,atol=1e-12):
        raise ValueError("Three nonnegative PCI weights must sum to 1")
    if not surfaces:
        raise ValueError("At least one surface is required")
    first=next(iter(surfaces.values()))
    for s in surfaces.values():
        if not (np.array_equal(s.moisture,first.moisture) and np.array_equal(s.density,first.density)
                and np.array_equal(s.mask,first.mask)):
            raise ValueError("Compared surfaces must have identical axes and domain masks")
    rows=[{"model_id":name,**components(s,**criteria)} for name,s in surfaces.items()]
    ref=max(r["roughness"] for r in rows)
    for r in rows:
        r["rough_ref"]=ref
        r["rough_norm"]=r["roughness"]/ref if ref else 0.0
        r["pci"]=float(a @ [r["mvr"],r["rough_norm"],r["pbv"]])
    return pd.DataFrame(rows)
