"""Utility functions for regression data handling and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler


def ensure_output_dir(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def read_tabular_csv(csv_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, sep=None, engine="python")
    except Exception as exc:
        raise ValueError(f"Could not read CSV file '{csv_path}': {exc}") from exc


def coerce_regression_target(target: pd.Series) -> pd.Series:
    numeric_target = pd.to_numeric(target, errors="coerce")
    if numeric_target.isna().any() and target.dtype == object:
        comma_decimal_target = pd.to_numeric(
            target.astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        if comma_decimal_target.isna().sum() < numeric_target.isna().sum():
            numeric_target = comma_decimal_target
    return numeric_target


def parse_feature_columns(
    feature_columns_arg: str,
    available_columns: list[str],
    target_column: str,
) -> list[str]:
    if not feature_columns_arg or not feature_columns_arg.strip():
        raise ValueError("--feature_columns cannot be empty. Use 'all' or a comma-separated column list.")

    if feature_columns_arg.strip().lower() == "all":
        selected_columns = [col for col in available_columns if col != target_column]
    else:
        selected_columns = []
        seen = set()
        for raw_column in feature_columns_arg.split(","):
            column = raw_column.strip()
            if not column:
                continue
            if column == target_column:
                raise ValueError(
                    f"Target column '{target_column}' cannot be included in --feature_columns."
                )
            if column not in available_columns:
                available = ", ".join(available_columns)
                raise ValueError(f"Feature column '{column}' was not found. Available columns: {available}")
            if column not in seen:
                selected_columns.append(column)
                seen.add(column)

    if not selected_columns:
        raise ValueError("--feature_columns did not select any feature columns.")
    return selected_columns


def select_features(df: pd.DataFrame, selected_feature_columns: list[str]) -> pd.DataFrame:
    missing_columns = [col for col in selected_feature_columns if col not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        available = ", ".join(df.columns)
        raise ValueError(f"Feature columns not found in CSV: {missing}. Available columns: {available}")
    return df[selected_feature_columns].copy()


def load_regression_dataset(
    data_path: str | Path,
    target_column: str,
    feature_columns: str = "all",
):
    csv_path = Path(data_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = read_tabular_csv(csv_path)
    if target_column not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(f"Target column '{target_column}' was not found. Available columns: {available}")

    original_rows = len(df)
    df = df.dropna(subset=[target_column]).reset_index(drop=True)
    dropped_missing_target = original_rows - len(df)

    y = coerce_regression_target(df[target_column])
    if y.isna().any():
        bad_count = int(y.isna().sum())
        raise ValueError(
            f"Regression target column '{target_column}' must be numeric. "
            f"Found {bad_count} non-numeric target values after dropping missing targets."
        )

    selected_feature_columns = parse_feature_columns(feature_columns, df.columns.tolist(), target_column)
    X = df[selected_feature_columns].copy()
    if X.shape[1] == 0:
        raise ValueError("The dataset must contain at least one feature column besides the target.")

    metadata = {
        "rows_loaded": int(original_rows),
        "rows_after_dropping_missing_target": int(len(df)),
        "dropped_missing_target_rows": int(dropped_missing_target),
        "n_features": int(X.shape[1]),
        "selected_feature_columns": selected_feature_columns,
    }
    return X, y.astype(float), metadata


def load_regression_split_dataset(
    data_path: str | Path,
    target_column: str,
    selected_feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    csv_path = Path(data_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = read_tabular_csv(csv_path)
    if target_column not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(f"Target column '{target_column}' was not found in {csv_path}. Available columns: {available}")

    original_rows = len(df)
    df = df.dropna(subset=[target_column]).reset_index(drop=True)
    dropped_missing_target = original_rows - len(df)

    y = coerce_regression_target(df[target_column])
    if y.isna().any():
        bad_count = int(y.isna().sum())
        raise ValueError(
            f"Regression target column '{target_column}' in {csv_path} must be numeric. "
            f"Found {bad_count} non-numeric target values after dropping missing targets."
        )

    X = select_features(df, selected_feature_columns)
    if X.shape[1] == 0:
        raise ValueError("The dataset must contain at least one selected feature column.")

    metadata = {
        "path": str(csv_path),
        "rows_loaded": int(original_rows),
        "rows_after_dropping_missing_target": int(len(df)),
        "dropped_missing_target_rows": int(dropped_missing_target),
        "n_features": int(X.shape[1]),
    }
    return X, y.astype(float), metadata


def load_regression_predefined_splits(
    train_data_path: str | Path,
    validation_data_path: str | Path,
    target_column: str,
    feature_columns: str = "all",
    conformal_data_path: str | Path | None = None,
):
    train_csv_path = Path(train_data_path)
    if not train_csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {train_csv_path}")

    train_df = read_tabular_csv(train_csv_path)
    if target_column not in train_df.columns:
        available = ", ".join(train_df.columns)
        raise ValueError(f"Target column '{target_column}' was not found in {train_csv_path}. Available columns: {available}")

    selected_feature_columns = parse_feature_columns(feature_columns, train_df.columns.tolist(), target_column)

    X_train, y_train, train_metadata = load_regression_split_dataset(
        train_data_path,
        target_column,
        selected_feature_columns,
    )
    X_cal = y_cal = None
    conformal_metadata = None
    if conformal_data_path is not None:
        X_cal, y_cal, conformal_metadata = load_regression_split_dataset(
            conformal_data_path,
            target_column,
            selected_feature_columns,
        )
    X_valid, y_valid, validation_metadata = load_regression_split_dataset(
        validation_data_path,
        target_column,
        selected_feature_columns,
    )

    if len(X_train) < 2:
        raise ValueError("The training CSV has fewer than 2 usable rows.")
    if X_cal is not None and len(X_cal) < 1:
        raise ValueError("The conformal CSV must contain at least one usable row.")
    if len(X_valid) < 1:
        raise ValueError("The validation CSV must contain at least one usable row.")

    metadata = {
        "input_mode": "predefined_csv_splits",
        "selected_feature_columns": selected_feature_columns,
        "train": train_metadata,
        "conformal": conformal_metadata,
        "validation": validation_metadata,
    }
    return X_train, X_cal, X_valid, y_train, y_cal, y_valid, metadata


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_ordinal_encoder() -> OrdinalEncoder:
    try:
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    except TypeError:
        return OrdinalEncoder()


def coerce_numeric_features(values):
    if isinstance(values, pd.DataFrame):
        return values.apply(
            lambda column: pd.to_numeric(
                column.astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
        )
    frame = pd.DataFrame(values)
    return frame.apply(
        lambda column: pd.to_numeric(
            column.astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
    )


def parse_optional_column_list(
    columns_arg: str | None,
    selected_columns: list[str],
    flag_name: str,
) -> list[str]:
    if columns_arg is None or not columns_arg.strip() or columns_arg.strip().lower() == "none":
        return []
    if columns_arg.strip().lower() == "all":
        return list(selected_columns)

    parsed_columns = []
    seen = set()
    for raw_column in columns_arg.split(","):
        column = raw_column.strip()
        if not column:
            continue
        if column not in selected_columns:
            available = ", ".join(selected_columns)
            raise ValueError(f"{flag_name} column '{column}' is not in --feature_columns. Selected columns: {available}")
        if column not in seen:
            parsed_columns.append(column)
            seen.add(column)
    return parsed_columns


def resolve_feature_types(
    X: pd.DataFrame,
    continuous_columns_arg: str | None = None,
    ohe_columns_arg: str | None = None,
    ordinal_columns_arg: str | None = None,
) -> dict[str, list[str]]:
    selected_columns = X.columns.tolist()
    continuous_columns = parse_optional_column_list(continuous_columns_arg, selected_columns, "--continuous_columns")
    ohe_columns = parse_optional_column_list(ohe_columns_arg, selected_columns, "--ohe_columns")
    ordinal_columns = parse_optional_column_list(ordinal_columns_arg, selected_columns, "--ordinal_columns")

    assigned_columns = continuous_columns + ohe_columns + ordinal_columns
    duplicated_columns = sorted({col for col in assigned_columns if assigned_columns.count(col) > 1})
    if duplicated_columns:
        duplicates = ", ".join(duplicated_columns)
        raise ValueError(f"Columns cannot appear in more than one preprocessing role: {duplicates}")

    unassigned_columns = [col for col in selected_columns if col not in set(assigned_columns)]
    auto_continuous_columns = X[unassigned_columns].select_dtypes(include=[np.number]).columns.tolist()
    auto_ohe_columns = [col for col in unassigned_columns if col not in auto_continuous_columns]

    return {
        "continuous_columns": continuous_columns + auto_continuous_columns,
        "ohe_columns": ohe_columns + auto_ohe_columns,
        "ordinal_columns": ordinal_columns,
        "auto_continuous_columns": auto_continuous_columns,
        "auto_ohe_columns": auto_ohe_columns,
    }


def build_preprocessor(
    X: pd.DataFrame,
    continuous_columns: str | None = None,
    ohe_columns: str | None = None,
    ordinal_columns: str | None = None,
) -> tuple[ColumnTransformer, dict[str, list[str]]]:
    feature_types = resolve_feature_types(X, continuous_columns, ohe_columns, ordinal_columns)

    transformers = []
    if feature_types["continuous_columns"]:
        numeric_pipeline = Pipeline(
            steps=[
                ("coerce_numeric", FunctionTransformer(coerce_numeric_features, validate=False)),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("continuous", numeric_pipeline, feature_types["continuous_columns"]))

    if feature_types["ohe_columns"]:
        ohe_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_one_hot_encoder()),
            ]
        )
        transformers.append(("one_hot", ohe_pipeline, feature_types["ohe_columns"]))

    if feature_types["ordinal_columns"]:
        ordinal_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", make_ordinal_encoder()),
            ]
        )
        transformers.append(("ordinal", ordinal_pipeline, feature_types["ordinal_columns"]))

    if not transformers:
        raise ValueError("No usable feature columns were detected.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    feature_info = {
        "numeric_columns": feature_types["continuous_columns"],
        "categorical_columns": feature_types["ohe_columns"],
        "continuous_columns": feature_types["continuous_columns"],
        "ohe_columns": feature_types["ohe_columns"],
        "ordinal_columns": feature_types["ordinal_columns"],
        "auto_continuous_columns": feature_types["auto_continuous_columns"],
        "auto_ohe_columns": feature_types["auto_ohe_columns"],
    }
    return preprocessor, feature_info


def split_train_calibration_test(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int,
    test_size: float = 0.2,
    calibration_size: float = 0.2,
):
    if len(X) < 5:
        raise ValueError("At least 5 rows are required to create train, calibration, and test splits.")

    X_train_cal, X_test, y_train_cal, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    relative_calibration_size = calibration_size / (1.0 - test_size)
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_cal,
        y_train_cal,
        test_size=relative_calibration_size,
        random_state=random_state,
    )

    if len(X_train) < 2:
        raise ValueError("The training split has fewer than 2 rows. Use a larger dataset.")
    if len(X_cal) < 1 or len(X_test) < 1:
        raise ValueError("Calibration and test splits must contain at least one row.")

    return X_train, X_cal, X_test, y_train, y_cal, y_test


def split_train_validation(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int,
    validation_size: float = 0.2,
):
    if len(X) < 3:
        raise ValueError("At least 3 rows are required to create train and validation splits.")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=random_state,
    )

    if len(X_train) < 2:
        raise ValueError("The training split has fewer than 2 rows. Use a larger dataset.")
    if len(X_valid) < 1:
        raise ValueError("The validation split must contain at least one row.")

    return X_train, X_valid, y_train, y_valid


def validate_common_args(cv_folds: int, alpha: float) -> None:
    if cv_folds < 2:
        raise ValueError("--cv_folds must be at least 2.")
    if not 0 < alpha < 1:
        raise ValueError("--alpha must be between 0 and 1.")


def effective_cv_folds(requested_folds: int, n_samples: int) -> int:
    folds = min(requested_folds, n_samples)
    if folds < 2:
        raise ValueError("At least 2 training rows are required for cross-validation.")
    return folds


def make_json_safe(value: Any):
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def save_json(data: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(data), f, indent=2)


def save_joblib(obj: Any, path: str | Path) -> None:
    joblib.dump(obj, path)


def save_predictions(predictions: pd.DataFrame, path: str | Path) -> None:
    predictions.to_csv(path, index=False)
