"""Plug in any fitted model exposing predict(X), without editing toolkit code."""
import numpy as np
from physical_consistency import Surface,generate_grid,compare_surfaces,write_surfaces


def evaluate_my_models(models, training_predictors, destination):
    X=np.asarray(training_predictors,dtype=float)
    template=generate_grid((X[:,0].min(),X[:,0].max()),(X[:,1].min(),X[:,1].max()),support=X)
    surfaces={name:Surface(template.moisture,template.density,
                          model.predict(template.points).reshape(template.z.shape),template.mask)
              for name,model in models.items()}
    write_surfaces(surfaces,destination)
    return compare_surfaces(surfaces)
