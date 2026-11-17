"""Independent holdout metrics and nondominated accuracy/PCI comparisons."""
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error


def accuracy(y_true,y_pred):
    a=np.asarray(y_true,dtype=float);b=np.asarray(y_pred,dtype=float)
    if a.ndim!=1 or a.shape!=b.shape or len(a)<2 or not np.isfinite([a,b]).all():
        raise ValueError("Accuracy requires >=2 finite paired one-dimensional values")
    mse=float(mean_squared_error(a,b))
    # Zero observations make MAPE undefined; do not replace zero by epsilon.
    return {"r2":float(r2_score(a,b)) if np.ptp(a)>0 else None,"mse":mse,
            "rmse":float(np.sqrt(mse)),"mae":float(mean_absolute_error(a,b)),
            "mape_pct":float(np.mean(np.abs((a-b)/a))*100) if np.all(a!=0) else None,
            "mean_residual":float(np.mean(b-a)),"n_test":len(a)}


def pareto_mask(points):
    """True for nondominated points; all coordinates are minimised; ties survive."""
    a=np.asarray(points,dtype=float)
    if a.ndim!=2 or not np.isfinite(a).all():
        raise ValueError("Pareto coordinates must be a finite two-dimensional array")
    return np.array([not np.any(np.all(a<=p,axis=1)&np.any(a<p,axis=1)) for p in a])
