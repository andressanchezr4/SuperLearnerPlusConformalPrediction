"""Model registry for the classification SuperLearner."""

from __future__ import annotations

import importlib
import inspect
from typing import Iterable

from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError:
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
    from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


AVAILABLE_MODELS = [
    "logistic_regression",
    "ridge_classifier",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "svm",
    "knn",
    "naive_bayes",
    "mlp",
    "xgboost_classifier",
    "lightgbm_classifier",
    "catboost_classifier",
]


OPTIONAL_MODEL_PACKAGES = {
    "xgboost_classifier": "xgboost",
    "lightgbm_classifier": "lightgbm",
    "catboost_classifier": "catboost",
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
            raise ValueError(f"Unknown classification model '{name}'. Available models: {available}")
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
    if estimator_cls is MLPClassifier:
        return {"log_loss"}
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
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=2000, solver="saga", random_state=random_state)
    if model_name == "ridge_classifier":
        return RidgeClassifier(solver="sag", random_state=random_state)
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    if model_name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state)
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=random_state)
    if model_name == "svm":
        return SVC(C=1.0, kernel="rbf", probability=True, random_state=random_state)
    if model_name == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    if model_name == "naive_bayes":
        return GaussianNB()
    if model_name == "mlp":
        kwargs = _mlp_kwargs_with_loss(
            MLPClassifier,
            {
                "hidden_layer_sizes": (100,),
                "max_iter": 500,
                "random_state": random_state,
            },
            mlp_loss,
            "log_loss",
            model_name,
        )
        return MLPClassifier(**kwargs)
    if model_name == "xgboost_classifier":
        xgboost = _require_package("xgboost", model_name)
        return xgboost.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
        )
    if model_name == "lightgbm_classifier":
        lightgbm = _require_package("lightgbm", model_name)
        return lightgbm.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "catboost_classifier":
        catboost = _require_package("catboost", model_name)
        return catboost.CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=random_state,
            verbose=False,
        )

    available = ", ".join(AVAILABLE_MODELS)
    raise ValueError(f"Unknown classification model '{model_name}'. Available models: {available}")


def create_models(
    model_names: Iterable[str],
    random_state: int,
    mlp_loss: str | None = None,
) -> list[tuple[str, object]]:
    validate_optional_dependencies(model_names)
    return [(name, create_model(name, random_state, mlp_loss)) for name in model_names]
