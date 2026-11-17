"""Estimator factory and training-only hyperparameter selection."""
import math
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, KFold, GroupKFold


def make_model(family, params=None, seed=42):
    """Return a scikit-learn estimator; params use native estimator parameter names."""
    params=dict(params or {})
    if family=="svr":
        defaults={"C":10.0,"epsilon":0.05};defaults.update(params)
        return make_pipeline(StandardScaler(),SVR(**defaults))
    if family=="knn":
        defaults={"n_neighbors":7,"weights":"distance"};defaults.update(params)
        return make_pipeline(StandardScaler(),KNeighborsRegressor(**defaults))
    if family=="dt":
        defaults={"max_depth":6,"min_samples_leaf":3,"random_state":seed};defaults.update(params)
        return DecisionTreeRegressor(**defaults)
    if family=="rf":
        defaults={"n_estimators":100,"min_samples_leaf":2,"random_state":seed,"n_jobs":1};defaults.update(params)
        return RandomForestRegressor(**defaults)
    if family=="xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ValueError("xgb requires installation with the [xgboost] extra") from exc
        defaults={"n_estimators":100,"max_depth":3,"learning_rate":0.05,
                  "random_state":seed,"n_jobs":1,"objective":"reg:squarederror"};defaults.update(params)
        return XGBRegressor(**defaults)
    raise ValueError(f"Unknown model family: {family}; choose svr, knn, dt, rf, xgb")


def fit_model(spec, X, y, *, seed=42, groups=None):
    """Fit only supplied training rows, including scaling and optional inner CV.

    search_grid accepts native parameter names; pipeline prefixes are supplied.
    Tuning optimises R2 only. PCI is never used to train or select parameters.
    """
    if set(spec)-{"id","family","params","search_grid","cv_folds"}:
        raise ValueError("Unknown model configuration option")
    model=make_model(spec["family"],spec.get("params"),seed)
    grid=spec.get("search_grid")
    info={"selected_params":model.get_params(deep=False),"cv_best_r2":None}
    if grid:
        folds=spec.get("cv_folds",5)
        if not isinstance(folds,int) or folds<2 or len(X)<2*folds:
            raise ValueError("CV requires >=2 folds with enough training rows for R2")
        prefix={"svr":"svr__","knn":"kneighborsregressor__"}.get(spec["family"],"")
        grid={prefix+k:v for k,v in grid.items()}
        if groups is None:
            cv=KFold(folds,shuffle=True,random_state=seed)
        else:
            if len(set(groups))<folds:
                raise ValueError("Insufficient training specimen groups for CV folds")
            cv=GroupKFold(folds)
        search=GridSearchCV(model,grid,scoring="r2",cv=cv,n_jobs=1,error_score="raise")
        search.fit(X,y,groups=groups)
        model=search.best_estimator_
        info={"selected_params":search.best_params_,"cv_best_r2":float(search.best_score_)}
    else:
        model.fit(X,y)
    # JSON-safe native parameters, without serialising fitted models or object reprs.
    native=model.steps[-1][1] if hasattr(model,"steps") else model
    info["fitted_estimator_params"]={k:v for k,v in native.get_params().items()
                                       if v is None or isinstance(v,(str,int,float,bool))}
    for key,value in info["fitted_estimator_params"].items():
        if isinstance(value,float) and not math.isfinite(value):
            info["fitted_estimator_params"][key]="NaN" if math.isnan(value) else ("Infinity" if value>0 else "-Infinity")
    info.pop("selected_params",None) if not grid else None
    return model,info
