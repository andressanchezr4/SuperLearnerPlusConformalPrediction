"""Train a tabular SuperLearner with split conformal prediction sets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classification.conformal_classification import (
    calibrate_conformal_classification,
    conformal_coverage,
    make_prediction_sets,
)
from classification.models import create_models, parse_model_names
from classification.progress import progress_bar, setup_logger, timed_step
from classification.utils import (
    build_preprocessor,
    effective_stratified_cv_folds,
    encode_target,
    ensure_output_dir,
    load_classification_predefined_splits,
    load_classification_dataset,
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
        description="Train a classification SuperLearner with conformal prediction sets."
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
            "For current scikit-learn versions without an MLP loss parameter, only 'log_loss' is accepted."
        ),
    )
    parser.add_argument("--output_dir", required=True, help="Directory for models and outputs.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage level for conformal sets.")
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


def fit_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    pipeline = adjust_model_for_sample_size(pipeline, len(X))
    return pipeline.fit(X, y)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -500, 500)
    return 1.0 / (1.0 + np.exp(-values))


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def _scores_to_probabilities(scores: np.ndarray, n_classes_seen: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        if n_classes_seen == 2:
            positive = _sigmoid(scores)
            return np.column_stack([1.0 - positive, positive])
        return np.ones((scores.shape[0], n_classes_seen), dtype=float) / n_classes_seen
    return _softmax(scores)


def _hard_predictions_to_probabilities(predictions: np.ndarray, estimator_classes: np.ndarray) -> np.ndarray:
    class_to_position = {label: idx for idx, label in enumerate(estimator_classes)}
    probabilities = np.zeros((predictions.shape[0], estimator_classes.shape[0]), dtype=float)
    for row_idx, label in enumerate(predictions):
        probabilities[row_idx, class_to_position[label]] = 1.0
    return probabilities


def _align_probabilities(
    probabilities: np.ndarray,
    estimator_classes: np.ndarray,
    all_classes: np.ndarray,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    estimator_classes = np.asarray(estimator_classes)

    aligned = np.zeros((probabilities.shape[0], all_classes.shape[0]), dtype=float)
    class_to_column = {label: idx for idx, label in enumerate(all_classes)}
    for source_idx, label in enumerate(estimator_classes):
        if label in class_to_column and source_idx < probabilities.shape[1]:
            aligned[:, class_to_column[label]] = probabilities[:, source_idx]

    row_sums = aligned.sum(axis=1, keepdims=True)
    zero_rows = row_sums.squeeze() == 0
    if np.any(zero_rows):
        aligned[zero_rows, :] = 1.0 / all_classes.shape[0]
        row_sums = aligned.sum(axis=1, keepdims=True)
    return aligned / row_sums


def _call_estimator_method_in_chunks(model, method_name: str, X: pd.DataFrame, chunk_size: int = 100) -> np.ndarray:
    method = getattr(model, method_name)
    if len(X) <= chunk_size:
        return np.asarray(method(X))

    outputs = []
    for start in range(0, len(X), chunk_size):
        stop = start + chunk_size
        X_chunk = X.iloc[start:stop] if hasattr(X, "iloc") else X[start:stop]
        outputs.append(np.asarray(method(X_chunk)))
    return np.concatenate(outputs, axis=0)


def predict_proba_aligned(model: Pipeline, X: pd.DataFrame, all_classes: np.ndarray) -> np.ndarray:
    final_estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model
    if hasattr(final_estimator, "predict_proba"):
        probabilities = _call_estimator_method_in_chunks(model, "predict_proba", X)
    elif hasattr(final_estimator, "decision_function"):
        scores = _call_estimator_method_in_chunks(model, "decision_function", X)
        estimator_classes = getattr(model, "classes_", all_classes)
        probabilities = _scores_to_probabilities(scores, len(estimator_classes))
    else:
        estimator_classes = getattr(model, "classes_", all_classes)
        hard_predictions = _call_estimator_method_in_chunks(model, "predict", X)
        probabilities = _hard_predictions_to_probabilities(hard_predictions, estimator_classes)

    estimator_classes = np.asarray(getattr(model, "classes_", all_classes))
    return _align_probabilities(probabilities, estimator_classes, all_classes)


def predict_meta_probabilities(meta_model: LogisticRegression, X: np.ndarray, all_classes: np.ndarray) -> np.ndarray:
    if hasattr(meta_model, "decision_function"):
        scores = _call_estimator_method_in_chunks(meta_model, "decision_function", X)
        probabilities = _scores_to_probabilities(scores, len(meta_model.classes_))
    else:
        probabilities = _call_estimator_method_in_chunks(meta_model, "predict_proba", X)
    return _align_probabilities(probabilities, meta_model.classes_, all_classes)


def make_cv_splitter(y_train: np.ndarray, requested_folds: int, random_state: int, logger):
    effective_folds = effective_stratified_cv_folds(requested_folds, y_train)
    _, counts = np.unique(y_train, return_counts=True)
    if counts.min() >= effective_folds:
        splitter = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=random_state)
        logger.info("Using StratifiedKFold with %d folds.", effective_folds)
        return splitter, effective_folds, True

    splitter = KFold(n_splits=effective_folds, shuffle=True, random_state=random_state)
    logger.info(
        "Using KFold with %d folds because at least one class has fewer than 2 training rows.",
        effective_folds,
    )
    return splitter, effective_folds, False


def build_oof_predictions(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_specs: list[tuple[str, object]],
    preprocessor,
    cv_folds: int,
    random_state: int,
    all_classes: np.ndarray,
    logger,
) -> tuple[np.ndarray, dict[str, list[float]], bool]:
    n_samples = len(X_train)
    n_classes = all_classes.shape[0]
    oof_predictions = np.zeros((n_samples, len(model_specs) * n_classes), dtype=float)
    fold_metrics: dict[str, list[float]] = {}
    splitter, effective_folds, used_stratified_cv = make_cv_splitter(y_train, cv_folds, random_state, logger)
    if effective_folds != cv_folds:
        logger.info("Using %d CV folds after checking class counts.", effective_folds)

    if used_stratified_cv:
        splits = list(splitter.split(X_train, y_train))
    else:
        splits = list(splitter.split(X_train))

    model_iterator = progress_bar(model_specs, desc="Base models", total=len(model_specs))
    for model_idx, (model_name, estimator) in enumerate(model_iterator):
        logger.info("Training OOF probabilities for base model: %s", model_name)
        fold_metrics[model_name] = []
        start_col = model_idx * n_classes
        end_col = start_col + n_classes
        split_iterator = progress_bar(
            splits,
            desc=f"{model_name} folds",
            total=effective_folds,
            leave=False,
        )
        for fold_idx, (train_idx, valid_idx) in enumerate(split_iterator, start=1):
            fold_start = time.perf_counter()
            pipeline = build_pipeline(preprocessor, estimator)
            try:
                fit_pipeline(pipeline, X_train.iloc[train_idx], y_train[train_idx])
                probabilities = predict_proba_aligned(pipeline, X_train.iloc[valid_idx], all_classes)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed while training classification model '{model_name}' on fold {fold_idx}. "
                    f"Original error: {exc}"
                ) from exc

            oof_predictions[valid_idx, start_col:end_col] = probabilities
            predictions = np.argmax(probabilities, axis=1)
            accuracy = float(accuracy_score(y_train[valid_idx], predictions))
            fold_metrics[model_name].append(accuracy)
            elapsed = time.perf_counter() - fold_start
            logger.info(
                "%s fold %d/%d accuracy=%.6f elapsed=%.2fs",
                model_name,
                fold_idx,
                effective_folds,
                accuracy,
                elapsed,
            )

    return oof_predictions, fold_metrics, used_stratified_cv


def fit_final_base_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
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
                f"Failed while fitting final classification model '{model_name}'. Original error: {exc}"
            ) from exc
    return final_models


def predict_meta_features(base_models: dict[str, Pipeline], X: pd.DataFrame, all_classes: np.ndarray) -> np.ndarray:
    return np.column_stack([predict_proba_aligned(model, X, all_classes) for model in base_models.values()])


def labels_for_prediction_sets(prediction_sets: list[list[int]], label_encoder) -> list[str]:
    labels = []
    for prediction_set in prediction_sets:
        if prediction_set:
            class_labels = label_encoder.inverse_transform(np.asarray(prediction_set, dtype=int)).tolist()
        else:
            class_labels = []
        labels.append(json.dumps(class_labels))
    return labels


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

        logger.info("Selected classification models: %s", ", ".join(model_names))
        logger.info("Input mode: %s", input_mode)
        logger.info("Conformal prediction enabled: %s", args.use_conformal)

        with timed_step(logger, "Loading data"):
            if input_mode == "predefined_csv_splits":
                (
                    X_train,
                    X_cal,
                    X_test,
                    y_train_raw,
                    y_cal_raw,
                    y_test_raw,
                    data_metadata,
                ) = load_classification_predefined_splits(
                    args.train_data_path,
                    args.validation_data_path,
                    args.target_column,
                    args.feature_columns,
                    args.conformal_data_path if args.use_conformal else None,
                )
                y_train, label_encoder = encode_target(y_train_raw)
                y_cal = label_encoder.transform(y_cal_raw) if y_cal_raw is not None else None
                y_test = label_encoder.transform(y_test_raw)
                split_info = {
                    "input_mode": input_mode,
                    "predefined_splits": True,
                }
            else:
                X, y_raw, data_metadata = load_classification_dataset(
                    args.data_path,
                    args.target_column,
                    args.feature_columns,
                )
                y, label_encoder = encode_target(y_raw)
                if args.use_conformal:
                    X_train, X_cal, X_test, y_train, y_cal, y_test, split_info = split_train_calibration_test(
                        X,
                        y,
                        random_state=args.random_state,
                    )
                    split_info["input_mode"] = input_mode
                    split_info["predefined_splits"] = False
                    split_info["automatic_split"] = "train_conformal_validation"
                else:
                    X_train, X_test, y_train, y_test, split_info = split_train_validation(
                        X,
                        y,
                        random_state=args.random_state,
                    )
                    X_cal = y_cal = None
                    split_info["input_mode"] = input_mode
                    split_info["predefined_splits"] = False
                    split_info["automatic_split"] = "train_validation"

            class_labels = label_encoder.classes_.tolist()
            all_classes = np.arange(len(class_labels))
            logger.info("Train rows: %d | Validation rows: %d", len(X_train), len(X_test))
            if args.use_conformal:
                logger.info("Conformal rows: %d", len(X_cal))
            logger.info("Feature columns: %d | Classes: %d", X_train.shape[1], len(class_labels))
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

        with timed_step(logger, "Generating out-of-fold probabilities"):
            oof_predictions, fold_metrics, used_stratified_cv = build_oof_predictions(
                X_train,
                y_train,
                model_specs,
                preprocessor,
                args.cv_folds,
                args.random_state,
                all_classes,
                logger,
            )

        with timed_step(logger, "Training classification meta-model"):
            meta_model = LogisticRegression(max_iter=2000, solver="saga", random_state=args.random_state)
            meta_model.fit(oof_predictions, y_train)

        with timed_step(logger, "Refitting final base models"):
            base_models = fit_final_base_models(X_train, y_train, model_specs, preprocessor, logger)

        q_hat = None
        calibration_scores = np.array([])
        if args.use_conformal:
            with timed_step(logger, "Calibrating conformal prediction sets"):
                logger.info("Building conformal meta-features...")
                cal_meta_features = predict_meta_features(base_models, X_cal, all_classes)
                logger.info("Predicting conformal probabilities...")
                cal_probabilities = predict_meta_probabilities(meta_model, cal_meta_features, all_classes)
                q_hat, calibration_scores = calibrate_conformal_classification(y_cal, cal_probabilities, alpha=args.alpha)
                logger.info("Conformal q_hat: %.6f", q_hat)
        else:
            logger.info("Conformal prediction disabled; skipping calibration.")

        with timed_step(logger, "Evaluating final SuperLearner"):
            logger.info("Building validation meta-features...")
            test_meta_features = predict_meta_features(base_models, X_test, all_classes)
            logger.info("Predicting validation probabilities...")
            test_probabilities = predict_meta_probabilities(meta_model, test_meta_features, all_classes)
            logger.info("Computing validation predictions and metrics...")
            test_predictions = np.argmax(test_probabilities, axis=1)
            if args.use_conformal:
                prediction_sets = make_prediction_sets(test_probabilities, q_hat)
                coverage = conformal_coverage(y_test, prediction_sets)
                mean_prediction_set_size = float(np.mean([len(prediction_set) for prediction_set in prediction_sets]))
            else:
                prediction_sets = [[] for _ in range(len(y_test))]
                coverage = None
                mean_prediction_set_size = None

            accuracy = float(accuracy_score(y_test, test_predictions))
            balanced_accuracy = float(balanced_accuracy_score(y_test, test_predictions))
            f1_macro = float(f1_score(y_test, test_predictions, average="macro", zero_division=0))
            ll = float(log_loss(y_test, test_probabilities, labels=all_classes))

            metrics = {
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "f1_macro": f1_macro,
                "log_loss": ll,
                "conformal_coverage": coverage,
                "mean_prediction_set_size": mean_prediction_set_size,
                "q_hat": q_hat,
                "fold_accuracy": fold_metrics,
            }
            logger.info(
                "Final accuracy=%.6f balanced_accuracy=%.6f f1_macro=%.6f log_loss=%.6f",
                accuracy,
                balanced_accuracy,
                f1_macro,
                ll,
            )
            if args.use_conformal:
                logger.info(
                    "Conformal coverage=%.6f mean prediction set size=%.6f",
                    coverage,
                    mean_prediction_set_size,
                )

        with timed_step(logger, "Saving artifacts"):
            y_test_labels = label_encoder.inverse_transform(y_test)
            y_pred_labels = label_encoder.inverse_transform(test_predictions)
            predicted_probability = np.max(test_probabilities, axis=1)
            predictions_df = pd.DataFrame(
                {
                    "y_true": y_test_labels,
                    "y_pred": y_pred_labels,
                    "predicted_probability": predicted_probability,
                    "conformal_prediction_set": labels_for_prediction_sets(prediction_sets, label_encoder),
                }
            )
            config = {
                "task": "classification",
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
                "class_labels": class_labels,
                "split_sizes": {
                    "train": len(X_train),
                    "conformal": len(X_cal) if X_cal is not None else None,
                    "calibration": len(X_cal) if X_cal is not None else None,
                    "validation": len(X_test),
                    "test": len(X_test),
                },
                "split_info": split_info,
                "used_stratified_cv": used_stratified_cv,
            }
            save_joblib(
                {
                    "meta_model": meta_model,
                    "base_model_names": list(base_models.keys()),
                    "label_encoder": label_encoder,
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
            logger.info("Calibration scores stored in memory only: %d values.", len(calibration_scores))
        return 0

    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
