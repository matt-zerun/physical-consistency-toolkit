"""Regenerate the independently specified synthetic examples in this directory."""
from pathlib import Path
import numpy as np
from physical_consistency import generate_synthetic,generate_grid,Surface,write_surfaces
from physical_consistency.synthetic import generate_synthetic_eda


def generate(destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    generate_synthetic().to_csv(destination/"synthetic.csv",index=False)
    generate_synthetic_eda().to_csv(destination/"synthetic_eda.csv",index=False)
    base=generate_grid((2,17),(1.35,1.75),shape=(21,19))
    w,d=np.meshgrid(base.moisture,base.density,indexing="ij")
    values={"constant":np.ones_like(w),"increasing":0.5+0.04*w+0.3*d,
            "decreasing":2.5-0.04*w-0.3*d,
            "oscillatory":1.5+0.2*np.sin(2*w)+0.1*np.sin(30*d),
            "checkerboard":1.5+0.2*(np.indices(w.shape).sum(axis=0)%2),
            "out_of_bounds":np.full_like(w,5.2)}
    write_surfaces({name:Surface(base.moisture,base.density,z) for name,z in values.items()},
                   destination/"synthetic_surfaces.csv")

if __name__=="__main__":generate(Path(__file__).resolve().parent)
