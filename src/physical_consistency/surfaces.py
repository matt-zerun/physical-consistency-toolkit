"""Regular two-dimensional grids, masks, and public long-form CSV exchange."""
from dataclasses import dataclass
import re
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError
from .data import FEATURES


def _axis(a):
    a = np.asarray(a, dtype=float)
    if a.ndim != 1 or a.size < 2 or not np.isfinite(a).all() or np.any(np.diff(a) <= 0):
        raise ValueError("Each axis needs >=2 finite, strictly increasing coordinates")
    if not np.allclose(np.diff(a), a[1]-a[0], rtol=1e-7, atol=1e-12):
        raise ValueError("Only regular axes are supported")
    return a


def safe_name(name):
    if not isinstance(name,str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}",name):
        raise ValueError("Model IDs must match [A-Za-z][A-Za-z0-9_-]{0,63}")
    return name


@dataclass
class Surface:
    """z[i,j] means lambda at moisture[i], density[j]; mask defines the domain."""
    moisture: np.ndarray
    density: np.ndarray
    z: np.ndarray
    mask: np.ndarray | None = None

    def __post_init__(self):
        self.moisture = _axis(self.moisture).copy()
        self.density = _axis(self.density).copy()
        self.z = np.asarray(self.z, dtype=float).copy()
        if self.z.shape != (len(self.moisture),len(self.density)):
            raise ValueError("Surface shape must be (n_moisture,n_density)")
        if self.mask is None:
            self.mask = np.ones(self.z.shape,dtype=bool)
        else:
            raw = np.asarray(self.mask)
            if raw.shape != self.z.shape or not np.isin(raw,[True,False,0,1]).all():
                raise ValueError("Mask must be a boolean array matching the surface")
            self.mask = raw.astype(bool)
        if not self.mask.any() or not np.isfinite(self.z[self.mask]).all():
            raise ValueError("Included surface values must be finite; mask must be nonempty")
        self.z[~self.mask] = np.nan

    @property
    def points(self):
        w,d = np.meshgrid(self.moisture,self.density,indexing="ij")
        return np.column_stack([w.ravel(),d.ravel()])


def generate_grid(moisture_range, density_range, shape=(41,41), support=None):
    """Return an empty-valued Surface template; optional convex hull of support.

    support must come from training predictors when used for model comparison.
    No target values are used in construction of a domain.
    """
    if len(shape)!=2 or any(not isinstance(n,(int,np.integer)) or n<2 for n in shape):
        raise ValueError("Grid shape must contain two integers >=2")
    if len(moisture_range)!=2 or len(density_range)!=2:
        raise ValueError("Grid ranges require two endpoints")
    w = _axis(np.linspace(*moisture_range,shape[0]))
    d = _axis(np.linspace(*density_range,shape[1]))
    surface = Surface(w,d,np.zeros(shape))
    if support is not None:
        support = np.asarray(support,dtype=float)
        if support.ndim!=2 or support.shape[1]!=2 or not np.isfinite(support).all():
            raise ValueError("Hull support must be a finite (n,2) array")
        scale = np.ptp(support,axis=0)
        if np.any(scale<=0):
            raise ValueError("Hull support must span both predictors")
        origin = support.min(axis=0)
        try:
            mask = Delaunay((support-origin)/scale).find_simplex((surface.points-origin)/scale)>=0
        except QhullError as exc:
            raise ValueError("Hull support is collinear or insufficient") from exc
        surface = Surface(w,d,np.zeros(shape),mask.reshape(shape))
    return surface


def write_surfaces(surfaces, path):
    """Export full rectangular coordinates, predicted values and explicit masks."""
    rows=[]
    for name,s in surfaces.items():
        safe_name(name)
        rows.append(pd.DataFrame({"model_id":name,FEATURES[0]:s.points[:,0],
                    FEATURES[1]:s.points[:,1],"prediction_w_mk":s.z.ravel(),
                    "inside_domain":s.mask.ravel()}))
    if not rows:
        raise ValueError("At least one surface is required")
    pd.concat(rows,ignore_index=True).to_csv(path,index=False)


def read_surfaces(path):
    """Read a complete regular lattice per model; never pivot-average duplicates."""
    f=pd.read_csv(path)
    required={"model_id",*FEATURES,"prediction_w_mk","inside_domain"}
    if not required.issubset(f.columns) or f.empty or f.model_id.isna().any():
        raise ValueError("Invalid surface schema or missing model_id")
    out={}
    for name,g in f.groupby("model_id",sort=True):
        safe_name(name)
        if g.duplicated(FEATURES).any():
            raise ValueError("Duplicate surface coordinates")
        for col in [*FEATURES,"prediction_w_mk"]:
            g=g.copy();g[col]=pd.to_numeric(g[col],errors="coerce")
        if not np.isfinite(g[FEATURES].to_numpy()).all():
            raise ValueError("Surface coordinates must be finite numeric values")
        w=np.sort(g[FEATURES[0]].unique()); d=np.sort(g[FEATURES[1]].unique())
        if len(g)!=len(w)*len(d):
            raise ValueError("Incomplete regular lattice: include excluded coordinates with mask=False")
        flags=g.inside_domain.astype(str).str.lower().map({"true":True,"false":False,"1":True,"0":False})
        if flags.isna().any():
            raise ValueError("inside_domain must contain true/false or 1/0")
        g=g.assign(inside_domain=flags).sort_values(FEATURES)
        out[name]=Surface(w,d,g.prediction_w_mk.to_numpy().reshape(len(w),len(d)),
                         g.inside_domain.to_numpy().reshape(len(w),len(d)))
    return out
