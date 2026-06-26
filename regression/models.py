"""Model registry for the regression SuperLearner."""

from __future__ import annotations

import importlib
import inspect
from typing import Iterable

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except ImportError:
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
    from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


AVAILABLE_MODELS = [
    "linear_model",
    "ridge",
    "lasso",
    "elastic_net",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "svm",
    "knn",
    "mlp",
    "xgboost_regressor",
    "lightgbm_regressor",
    "catboost_regressor",
]


OPTIONAL_MODEL_PACKAGES = {
    "xgboost_regressor": "xgboost",
    "lightgbm_regressor": "lightgbm",
    "catboost_regressor": "catboost",
}


def parse_model_names(models_arg: str) -> list[str]:
    if not models_arg or not models_arg.strip():
        raise ValueError("--models cannot be empty. Use 'all' or a comma-separated model list.")

    if models_arg.strip().lower() == "all":
        return list(AVAILABLE_MODELS)

    names = []
    seen = set()
    for raw_name in models_arg.split(","):
        name = raw_name.strip()
        if not name:
            continue
        if name not in AVAILABLE_MODELS:
            available = ", ".join(AVAILABLE_MODELS)
            raise ValueError(f"Unknown regression model '{name}'. Available models: {available}")
        if name not in seen:
            names.append(name)
            seen.add(name)

    if not names:
        raise ValueError("--models did not contain any valid model names.")
    return names


def _require_package(package_name: str, model_name: str):
    try:
        return importlib.import_module(package_name)
    except ImportError as exc:
        raise ImportError(
            f"Model '{model_name}' requires the optional package '{package_name}', "
            "but it is not installed. Install optional model libraries with:\n"
            "pip install xgboost lightgbm catboost"
        ) from exc


def validate_optional_dependencies(model_names: Iterable[str]) -> None:
    for model_name in model_names:
        package_name = OPTIONAL_MODEL_PACKAGES.get(model_name)
        if package_name:
            _require_package(package_name, model_name)


def _normalise_mlp_loss(mlp_loss: str | None) -> str | None:
    if mlp_loss is None:
        return None
    value = mlp_loss.strip()
    if not value or value.lower() in {"none", "default"}:
        return None
    return value


def _allowed_mlp_losses(estimator_cls) -> set[str]:
    if estimator_cls is MLPRegressor:
        return {"squared_error", "poisson"}
    return set()


def _mlp_kwargs_with_loss(estimator_cls, kwargs: dict, mlp_loss: str | None, default_loss: str, model_name: str) -> dict:
    loss = _normalise_mlp_loss(mlp_loss)
    if loss is None:
        return kwargs

    allowed_losses = _allowed_mlp_losses(estimator_cls)
    if allowed_losses and loss not in allowed_losses:
        allowed = ", ".join(sorted(allowed_losses))
        raise ValueError(f"--mlp_loss '{loss}' is not valid for '{model_name}'. Allowed values: {allowed}")

    supports_loss = "loss" in inspect.signature(estimator_cls).parameters
    if supports_loss:
        kwargs["loss"] = loss
        return kwargs

    if loss == default_loss:
        return kwargs

    raise ValueError(
        f"--mlp_loss '{loss}' was requested for '{model_name}', but this scikit-learn version "
        f"does not expose a loss parameter for {estimator_cls.__name__}. "
        f"Use the default '{default_loss}' or upgrade scikit-learn."
    )


def create_model(model_name: str, random_state: int, mlp_loss: str | None = None):
    if model_name == "linear_model":
        return LinearRegression()
    if model_name == "ridge":
        return Ridge(alpha=1.0, solver="lsqr")
    if model_name == "lasso":
        return Lasso(alpha=0.001, max_iter=10000, random_state=random_state)
    if model_name == "elastic_net":
        return ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=random_state)
    if model_name == "random_forest":
        return RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    if model_name == "extra_trees":
        return ExtraTreesRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    if model_name == "gradient_boosting":
        return GradientBoostingRegressor(random_state=random_state)
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(random_state=random_state)
    if model_name == "svm":
        return SVR(C=1.0, epsilon=0.1)
    if model_name == "knn":
        return KNeighborsRegressor(n_neighbors=5)
    if model_name == "mlp":
        kwargs = _mlp_kwargs_with_loss(
            MLPRegressor,
            {
                "hidden_layer_sizes": (100,),
                "max_iter": 500,
                "random_state": random_state,
            },
            mlp_loss,
            "squared_error",
            model_name,
        )
        return MLPRegressor(**kwargs)
    if model_name == "xgboost_regressor":
        xgboost = _require_package("xgboost", model_name)
        return xgboost.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "lightgbm_regressor":
        lightgbm = _require_package("lightgbm", model_name)
        return lightgbm.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "catboost_regressor":
        catboost = _require_package("catboost", model_name)
        return catboost.CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=random_state,
            verbose=False,
        )

    available = ", ".join(AVAILABLE_MODELS)
    raise ValueError(f"Unknown regression model '{model_name}'. Available models: {available}")


def create_models(
    model_names: Iterable[str],
    random_state: int,
    mlp_loss: str | None = None,
) -> list[tuple[str, object]]:
    validate_optional_dependencies(model_names)
    return [(name, create_model(name, random_state, mlp_loss)) for name in model_names]
