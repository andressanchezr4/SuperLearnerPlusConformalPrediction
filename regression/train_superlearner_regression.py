"""Train a tabular SuperLearner with split conformal intervals for regression."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regression.conformal_regression import (
    calibrate_conformal_regression,
    conformal_coverage,
    make_prediction_intervals,
)
from regression.models import create_models, parse_model_names
from regression.progress import progress_bar, setup_logger, timed_step
from regression.utils import (
    build_preprocessor,
    effective_cv_folds,
    ensure_output_dir,
    load_regression_predefined_splits,
    load_regression_dataset,
    save_joblib,
    save_json,
    save_predictions,
    split_train_calibration_test,
    split_train_validation,
    validate_common_args,
)


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value: true/false, yes/no, or 1/0.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a regression SuperLearner with conformal prediction intervals."
    )
    parser.add_argument("--data_path", help="Path to one input CSV file. The script will create splits.")
    parser.add_argument("--train_data_path", help="Path to a pre-split training CSV file.")
    parser.add_argument("--validation_data_path", help="Path to a pre-split validation/evaluation CSV file.")
    parser.add_argument("--conformal_data_path", help="Path to a pre-split conformal calibration CSV file.")
    parser.add_argument("--target_column", required=True, help="Name of the target column.")
    parser.add_argument(
        "--feature_columns",
        default="all",
        help="Feature columns to include, as a comma-separated list, or 'all'.",
    )
    parser.add_argument(
        "--continuous_columns",
        help="Columns to treat as continuous numeric features. Unspecified columns are auto-detected.",
    )
    parser.add_argument(
        "--ohe_columns",
        help="Columns to treat as categorical features with one-hot encoding.",
    )
    parser.add_argument(
        "--ordinal_columns",
        help="Discrete columns to encode ordinally instead of one-hot encoding.",
    )
    parser.add_argument("--cv_folds", required=True, type=int, help="Number of CV folds.")
    parser.add_argument("--models", required=True, help="Comma-separated base models or 'all'.")
    parser.add_argument(
        "--mlp_loss",
        help=(
            "Loss for the MLP model. Default behavior uses scikit-learn's default. "
            "For current scikit-learn versions without an MLP loss parameter, only 'squared_error' is accepted."
        ),
    )
    parser.add_argument("--output_dir", required=True, help="Directory for models and outputs.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage level for conformal intervals.")
    parser.add_argument(
        "--use_conformal",
        type=str_to_bool,
        default=True,
        help="Whether to run conformal prediction. Use true or false. Default: true.",
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def build_pipeline(preprocessor, estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", clone(estimator)),
        ]
    )


def adjust_model_for_sample_size(pipeline: Pipeline, n_samples: int) -> Pipeline:
    params = pipeline.get_params()
    if "model__n_neighbors" in params:
        pipeline.set_params(model__n_neighbors=max(1, min(params["model__n_neighbors"], n_samples)))
    return pipeline


def fit_pipeline(pipeline: Pipeline, X: pd.DataFrame, y) -> Pipeline:
    pipeline = adjust_model_for_sample_size(pipeline, len(X))
    return pipeline.fit(X, y)


def build_oof_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_specs: list[tuple[str, object]],
    preprocessor,
    cv_folds: int,
    random_state: int,
    logger,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    n_samples = len(X_train)
    effective_folds = effective_cv_folds(cv_folds, n_samples)
    if effective_folds != cv_folds:
        logger.info("Using %d CV folds because the training split has %d rows.", effective_folds, n_samples)

    oof_predictions = np.zeros((n_samples, len(model_specs)), dtype=float)
    fold_metrics: dict[str, list[float]] = {}
    cv = KFold(n_splits=effective_folds, shuffle=True, random_state=random_state)

    model_iterator = progress_bar(model_specs, desc="Base models", total=len(model_specs))
    for model_idx, (model_name, estimator) in enumerate(model_iterator):
        logger.info("Training OOF predictions for base model: %s", model_name)
        fold_metrics[model_name] = []
        split_iterator = progress_bar(
            list(cv.split(X_train)),
            desc=f"{model_name} folds",
            total=effective_folds,
            leave=False,
        )
        for fold_idx, (train_idx, valid_idx) in enumerate(split_iterator, start=1):
            fold_start = time.perf_counter()
            pipeline = build_pipeline(preprocessor, estimator)
            try:
                fit_pipeline(pipeline, X_train.iloc[train_idx], y_train.iloc[train_idx])
                predictions = pipeline.predict(X_train.iloc[valid_idx])
            except Exception as exc:
                raise RuntimeError(
                    f"Failed while training regression model '{model_name}' on fold {fold_idx}. "
                    f"Original error: {exc}"
                ) from exc

            oof_predictions[valid_idx, model_idx] = predictions
            rmse = float(np.sqrt(mean_squared_error(y_train.iloc[valid_idx], predictions)))
            fold_metrics[model_name].append(rmse)
            elapsed = time.perf_counter() - fold_start
            logger.info("%s fold %d/%d RMSE=%.6f elapsed=%.2fs", model_name, fold_idx, effective_folds, rmse, elapsed)

    return oof_predictions, fold_metrics


def fit_final_base_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_specs: list[tuple[str, object]],
    preprocessor,
    logger,
) -> dict[str, Pipeline]:
    final_models = {}
    for model_name, estimator in progress_bar(model_specs, desc="Final base model fit", total=len(model_specs)):
        logger.info("Fitting final base model on full training data: %s", model_name)
        pipeline = build_pipeline(preprocessor, estimator)
        try:
            final_models[model_name] = fit_pipeline(pipeline, X_train, y_train)
        except Exception as exc:
            raise RuntimeError(
                f"Failed while fitting final regression model '{model_name}'. Original error: {exc}"
            ) from exc
    return final_models


def predict_meta_features(base_models: dict[str, Pipeline], X: pd.DataFrame) -> np.ndarray:
    return np.column_stack([model.predict(X) for model in base_models.values()])


def validate_input_mode(args: argparse.Namespace) -> str:
    predefined_paths = [args.train_data_path, args.validation_data_path, args.conformal_data_path]
    using_predefined_splits = any(path is not None for path in predefined_paths)

    if args.data_path and using_predefined_splits:
        raise ValueError(
            "Use either --data_path for automatic splitting or the pre-split CSV flags, not both."
        )
    if not args.data_path and not using_predefined_splits:
        raise ValueError(
            "Provide --data_path, or provide --train_data_path and --validation_data_path "
            "with optional --conformal_data_path."
        )

    if using_predefined_splits:
        if not args.train_data_path or not args.validation_data_path:
            raise ValueError("--train_data_path and --validation_data_path are required for pre-split CSV mode.")
        if args.use_conformal and not args.conformal_data_path:
            raise ValueError("--conformal_data_path is required when --use_conformal true in pre-split CSV mode.")
        return "predefined_csv_splits"

    return "single_csv_auto_split"


def main() -> int:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    logger = setup_logger(output_dir)

    try:
        validate_common_args(args.cv_folds, args.alpha)
        model_names = parse_model_names(args.models)

        input_mode = validate_input_mode(args)

        logger.info("Selected regression models: %s", ", ".join(model_names))
        logger.info("Input mode: %s", input_mode)
        logger.info("Conformal prediction enabled: %s", args.use_conformal)

        with timed_step(logger, "Loading data"):
            if input_mode == "predefined_csv_splits":
                X_train, X_cal, X_test, y_train, y_cal, y_test, data_metadata = load_regression_predefined_splits(
                    args.train_data_path,
                    args.validation_data_path,
                    args.target_column,
                    args.feature_columns,
                    args.conformal_data_path if args.use_conformal else None,
                )
                split_info = {
                    "input_mode": input_mode,
                    "predefined_splits": True,
                }
            else:
                X, y, data_metadata = load_regression_dataset(
                    args.data_path,
                    args.target_column,
                    args.feature_columns,
                )
                if args.use_conformal:
                    X_train, X_cal, X_test, y_train, y_cal, y_test = split_train_calibration_test(
                        X,
                        y,
                        random_state=args.random_state,
                    )
                    split_info = {
                        "input_mode": input_mode,
                        "predefined_splits": False,
                        "automatic_split": "train_conformal_validation",
                    }
                else:
                    X_train, X_test, y_train, y_test = split_train_validation(
                        X,
                        y,
                        random_state=args.random_state,
                    )
                    X_cal = y_cal = None
                    split_info = {
                        "input_mode": input_mode,
                        "predefined_splits": False,
                        "automatic_split": "train_validation",
                    }

            logger.info("Train rows: %d | Validation rows: %d", len(X_train), len(X_test))
            if args.use_conformal:
                logger.info("Conformal rows: %d", len(X_cal))
            logger.info("Selected feature columns: %s", ", ".join(data_metadata["selected_feature_columns"]))

        with timed_step(logger, "Building preprocessing pipeline"):
            preprocessor, feature_info = build_preprocessor(
                X_train,
                args.continuous_columns,
                args.ohe_columns,
                args.ordinal_columns,
            )
            logger.info("Numeric columns: %d", len(feature_info["numeric_columns"]))
            logger.info("Categorical columns: %d", len(feature_info["categorical_columns"]))
            logger.info("Ordinal columns: %d", len(feature_info["ordinal_columns"]))

        with timed_step(logger, "Creating model registry"):
            model_specs = create_models(model_names, args.random_state, args.mlp_loss)

        with timed_step(logger, "Generating out-of-fold predictions"):
            oof_predictions, fold_metrics = build_oof_predictions(
                X_train,
                y_train,
                model_specs,
                preprocessor,
                args.cv_folds,
                args.random_state,
                logger,
            )

        with timed_step(logger, "Training regression meta-model"):
            meta_model = Ridge(alpha=1.0, solver="lsqr")
            meta_model.fit(oof_predictions, y_train)

        with timed_step(logger, "Refitting final base models"):
            base_models = fit_final_base_models(X_train, y_train, model_specs, preprocessor, logger)

        q_hat = None
        calibration_residuals = np.array([])
        if args.use_conformal:
            with timed_step(logger, "Calibrating conformal intervals"):
                cal_meta_features = predict_meta_features(base_models, X_cal)
                cal_predictions = meta_model.predict(cal_meta_features)
                q_hat, calibration_residuals = calibrate_conformal_regression(y_cal, cal_predictions, alpha=args.alpha)
                logger.info("Conformal q_hat: %.6f", q_hat)
        else:
            logger.info("Conformal prediction disabled; skipping calibration.")

        with timed_step(logger, "Evaluating final SuperLearner"):
            test_meta_features = predict_meta_features(base_models, X_test)
            test_predictions = meta_model.predict(test_meta_features)
            if args.use_conformal:
                lower_bound, upper_bound = make_prediction_intervals(test_predictions, q_hat)
                coverage = conformal_coverage(y_test, lower_bound, upper_bound)
                mean_interval_width = float(np.mean(upper_bound - lower_bound))
            else:
                lower_bound = np.full_like(test_predictions, np.nan, dtype=float)
                upper_bound = np.full_like(test_predictions, np.nan, dtype=float)
                coverage = None
                mean_interval_width = None

            mae = float(mean_absolute_error(y_test, test_predictions))
            rmse = float(np.sqrt(mean_squared_error(y_test, test_predictions)))
            r2 = float(r2_score(y_test, test_predictions)) if len(y_test) > 1 else float("nan")

            metrics = {
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "conformal_coverage": coverage,
                "mean_interval_width": mean_interval_width,
                "q_hat": q_hat,
                "fold_rmse": fold_metrics,
            }
            logger.info("Final MAE=%.6f RMSE=%.6f R2=%.6f", mae, rmse, r2)
            if args.use_conformal:
                logger.info("Conformal coverage=%.6f mean interval width=%.6f", coverage, mean_interval_width)

        with timed_step(logger, "Saving artifacts"):
            predictions_df = pd.DataFrame(
                {
                    "y_true": np.asarray(y_test),
                    "y_pred": test_predictions,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                }
            )
            config = {
                "task": "regression",
                "input_mode": input_mode,
                "data_path": str(Path(args.data_path)) if args.data_path else None,
                "train_data_path": str(Path(args.train_data_path)) if args.train_data_path else None,
                "validation_data_path": str(Path(args.validation_data_path)) if args.validation_data_path else None,
                "conformal_data_path": str(Path(args.conformal_data_path)) if args.conformal_data_path else None,
                "target_column": args.target_column,
                "feature_columns": args.feature_columns,
                "continuous_columns": args.continuous_columns,
                "ohe_columns": args.ohe_columns,
                "ordinal_columns": args.ordinal_columns,
                "use_conformal": args.use_conformal,
                "cv_folds": args.cv_folds,
                "models": model_names,
                "mlp_loss": args.mlp_loss,
                "alpha": args.alpha,
                "random_state": args.random_state,
                "data": data_metadata,
                "features": feature_info,
                "split_sizes": {
                    "train": len(X_train),
                    "conformal": len(X_cal) if X_cal is not None else None,
                    "calibration": len(X_cal) if X_cal is not None else None,
                    "validation": len(X_test),
                    "test": len(X_test),
                },
                "split_info": split_info,
            }
            save_joblib(
                {
                    "meta_model": meta_model,
                    "base_model_names": list(base_models.keys()),
                    "use_conformal": args.use_conformal,
                    "alpha": args.alpha,
                    "q_hat": q_hat,
                },
                output_dir / "trained_superlearner.joblib",
            )
            save_joblib(base_models, output_dir / "base_models.joblib")
            save_json(metrics, output_dir / "metrics.json")
            save_json(config, output_dir / "config.json")
            save_predictions(predictions_df, output_dir / "predictions.csv")

        logger.info("Training finished. Artifacts saved in: %s", output_dir)
        if args.use_conformal:
            logger.info("Calibration residuals stored in memory only: %d values.", len(calibration_residuals))
        return 0

    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
