# Decision Log

## 2026-06-24

Context: Initial hackathon scaffold for a tabular SuperLearner with conformal prediction.

Decision/change: Created separate `regression/` and `classification/` pipelines with model registries, preprocessing utilities, conformal calibration modules, progress/logging helpers, CLI training scripts, `requirements.txt`, and `README.md`.

Rationale: Keep the code executable and easy to modify during a hackathon while preserving clear task separation between regression and classification.

Affected files: `regression/*`, `classification/*`, `README.md`, `requirements.txt`.

Verification: Pending smoke tests after the scaffold is compiled.

Next step: Run syntax checks and small end-to-end trainings with lightweight models.

## 2026-06-24

Context: Workspace includes `AGENTS.md`, which requires a dated technical decision log.

Decision/change: Added `docs/decision_log.md` and updated CSV loading to auto-detect delimiters.

Rationale: The local `data.csv` uses semicolon delimiters, and generic tabular CSV support is more useful for the requested hackathon skeleton than assuming comma-only files.

Affected files: `docs/decision_log.md`, `regression/utils.py`, `classification/utils.py`.

Verification: Pending syntax checks and smoke tests.

Next step: Compile Python modules and run regression/classification CLI smoke tests.

## 2026-06-24

Context: Smoke tests were run with the Anaconda Python available on the machine.

Decision/change: Added compatibility imports for `HistGradientBoostingRegressor` and `HistGradientBoostingClassifier`.

Rationale: Older scikit-learn versions require enabling the experimental histogram gradient boosting module before importing those estimators.

Affected files: `regression/models.py`, `classification/models.py`.

Verification: Initial smoke tests reached model imports and exposed the compatibility issue.

Next step: Re-run syntax checks and smoke tests.

## 2026-06-24

Context: Regression smoke fixture generated on Windows used comma decimal values such as `40,5`.

Decision/change: Added tolerant regression target coercion that retries decimal-comma parsing when normal numeric parsing leaves invalid values.

Rationale: Keeps CSV ingestion more robust for common locale-specific exports while preserving a clear error for genuinely non-numeric regression targets.

Affected files: `regression/utils.py`.

Verification: Pending re-run of regression smoke test.

Next step: Re-run compile and regression CLI smoke test.

## 2026-06-24

Context: Final verification after compatibility and CSV parsing fixes.

Decision/change: Ran syntax checks and smoke tests for both CLIs using lightweight model selections.

Rationale: Verify that the generated project is executable end to end without requiring optional gradient boosting packages.

Affected files: No code changes from this verification step.

Verification: `python -m compileall -q regression classification` passed. Classification smoke test passed with `logistic_regression,knn`; regression smoke test passed with `ridge,knn`. Outputs were written under the system temp directory.

Next step: Use `pip install -r requirements.txt` or an existing environment with the listed packages before running on the full dataset.

## 2026-06-24

Context: Classification with very rare classes can create folds where a base model sees only one class.

Decision/change: Added early validation that the training split contains every target class and at least two rows per class.

Rationale: Produces a clear, actionable error instead of estimator-specific failures during cross-validation.

Affected files: `classification/utils.py`.

Verification: Pending final compile and smoke tests.

Next step: Re-run syntax checks and classification smoke test.

## 2026-06-24

Context: Final verification after rare-class validation.

Decision/change: Re-ran syntax and classification smoke checks.

Rationale: Confirm that the added validation does not break a normal balanced classification run.

Affected files: No code changes from this verification step.

Verification: `python -m compileall -q regression classification` passed. Classification smoke test passed with `logistic_regression,knn`.

Next step: Run the project on the target hackathon dataset with the desired model list.

## 2026-06-24

Context: Pilot classification run requested on `data.csv` with `gutper` as target.

Decision/change: Ran `classification/train_superlearner_classification.py` with `logistic_regression,knn,naive_bayes`, `cv_folds=3`, and `output_dir=results/classification_pilot`.

Rationale: Use lightweight models for a fast functional pilot before trying heavier ensembles or optional libraries.

Affected files: Generated `results/classification_pilot/*`.

Verification: Run completed successfully. Test metrics: accuracy `0.9397515527950311`, balanced accuracy `0.9073538826243764`, f1 macro `0.9157758603487506`, log loss `0.13152764925833155`, conformal coverage `0.9298136645962732`, mean prediction set size `0.9826086956521739`.

Next step: For a modeling-quality run, consider removing or featurizing identifier-like columns (`hmdb_id`, `name`, `inchi`) because one-hot encoding them creates a large artifact and may encourage memorization.

## 2026-06-24

Context: User requested a flag for regression and classification to choose which feature columns are included, with `all` meaning all non-target columns.

Decision/change: Added optional `--feature_columns` to both training scripts. The default is `all`; comma-separated lists are validated against CSV columns; the target column is rejected if included.

Rationale: Allows excluding identifier-like or high-cardinality columns without editing the CSV or code, and prevents accidental target leakage.

Affected files: `regression/utils.py`, `classification/utils.py`, `regression/train_superlearner_regression.py`, `classification/train_superlearner_classification.py`, `README.md`.

Verification: `python -m compileall -q regression classification` passed. Regression smoke test passed with `--feature_columns x_num,ccl`. Classification pilot on `data.csv` passed with `--feature_columns ccl,icl`.

Next step: Prefer `--feature_columns ccl,icl` for the current pilot dataset unless molecular string/id columns are intentionally featurized.

## 2026-06-24

Context: Pilot comparison after adding feature selection.

Decision/change: Ran classification pilot on `data.csv` with only `ccl,icl`.

Rationale: Avoid one-hot encoding high-cardinality identifier-like fields (`hmdb_id`, `name`, `inchi`) while testing the new flag on the real pilot CSV.

Affected files: Generated `results/classification_pilot_selected_columns/*`.

Verification: Run completed successfully. Test metrics: accuracy `0.9391304347826087`, balanced accuracy `0.9060618929602937`, f1 macro `0.9148286099865047`, log loss `0.1313157833558611`, conformal coverage `0.9298136645962732`, mean prediction set size `0.9832298136645963`. `base_models.joblib` size was about `0.84 MB`.

Next step: If using `inchi`, add a chemistry-aware featurization instead of raw one-hot encoding.

## 2026-06-24

Context: User requested support for externally preprocessed train, validation, and conformal CSVs, plus an option to disable conformal prediction.

Decision/change: Added input modes for a single CSV with automatic splitting and for predefined CSV splits using `--train_data_path`, `--validation_data_path`, and `--conformal_data_path`. Added `--use_conformal true/false`.

Rationale: Allows controlled experiments with external preprocessing/splitting, while still preserving the simple one-CSV path for hackathon use.

Affected files: `regression/utils.py`, `classification/utils.py`, `regression/train_superlearner_regression.py`, `classification/train_superlearner_classification.py`, `README.md`.

Verification: Single-CSV classification and regression smoke tests passed with `--use_conformal false`. Predefined-split classification and regression smoke tests passed with `--use_conformal true`.

Next step: Use stratified or otherwise representative external splits for meaningful metrics; the tiny smoke splits are only execution checks.

## 2026-06-24

Context: User requested flags to explicitly specify whether columns are continuous or discrete, and for discrete columns whether one-hot encoding is needed.

Decision/change: Added `--continuous_columns`, `--ohe_columns`, and `--ordinal_columns` to both training scripts. Continuous columns are numerically coerced, median-imputed, and scaled. OHE columns use `OneHotEncoder`. Ordinal columns use `OrdinalEncoder`.

Rationale: Makes preprocessing choices explicit and reproducible, while preserving automatic type detection when no typing flags are provided.

Affected files: `regression/utils.py`, `classification/utils.py`, `regression/train_superlearner_regression.py`, `classification/train_superlearner_classification.py`, `README.md`.

Verification: Compile passed. Classification pilot passed with `--continuous_columns none --ohe_columns ccl,icl --ordinal_columns none`; predefined-split smoke tests passed with mixed OHE/ordinal typing.

Next step: For high-cardinality chemistry fields, prefer explicit feature engineering before listing them in `--ohe_columns`.

## 2026-06-24

Context: User requested README examples including all available training flags and explanations.

Decision/change: Added complete README examples for single-CSV classification, predefined-split classification, and predefined-split regression without conformal. Added a flag reference table and rules for mutually exclusive input modes and column typing.

Rationale: Make the scripts usable without reading the source code.

Affected files: `README.md`.

Verification: README now documents all current CLI flags shared by regression and classification.

Next step: Keep README in sync if future CLI flags are added.

## 2026-06-24

Context: User requested MLP support in both regression and classification, plus a flag to configure the MLP loss.

Decision/change: Added `mlp` to regression models using `MLPRegressor`. Kept classification `mlp` using `MLPClassifier`. Added `--mlp_loss` to both training scripts and passed it to the model registry. The registry only passes `loss` when the installed scikit-learn exposes that parameter; otherwise it accepts the estimator default (`squared_error` for regression, `log_loss` for classification) and errors clearly for unsupported custom losses.

Rationale: Keeps the feature usable on older scikit-learn versions while allowing newer versions to use configurable MLP losses.

Affected files: `regression/models.py`, `classification/models.py`, `regression/train_superlearner_regression.py`, `classification/train_superlearner_classification.py`, `README.md`.

Verification: `python -m compileall -q regression classification` passed. Classification smoke test passed with `--models mlp --mlp_loss log_loss`. Regression smoke test passed with `--models mlp --mlp_loss squared_error`; scikit-learn emitted a convergence warning, but artifacts were generated.

Next step: For serious MLP runs, consider adding CLI flags for MLP architecture and `max_iter` if the hackathon workflow needs tuning.

## 2026-06-25

Context: User requested a Conda environment named `hackaton` with all required project libraries.

Decision/change: Created Conda environment `hackaton` with Python 3.10 and the dependencies from `requirements.txt`: `pandas`, `numpy`, `scikit-learn`, `tqdm`, `joblib`, `xgboost`, `lightgbm`, and `catboost`.

Rationale: Provide a reproducible environment for running the SuperLearner training scripts without relying on the base Anaconda environment.

Affected files: No project code changes. Environment created at `C:\Users\andres.sanchez\.conda\envs\hackaton`.

Verification: Imports succeeded with `C:\Users\andres.sanchez\.conda\envs\hackaton\python.exe`. Versions verified: Python `3.10.20`, pandas `2.3.3`, numpy `2.2.6`, scikit-learn `1.7.2`, xgboost `3.2.0`, lightgbm `4.6.0`, catboost `1.2.10`. `python -m compileall -q regression classification` passed in the environment.

Next step: Run training commands via `conda activate hackaton` or direct interpreter path.

## 2026-06-25

Context: Code review found inconsistencies after adding column typing, MLP loss, optional conformal prediction, and the new Conda environment.

Decision/change: Added package-qualified imports and `__init__.py` files so saved `joblib` artifacts can be loaded from the project root. Validated `--mlp_loss` early for regression and classification. Updated README examples and model lists for `mlp` and `--mlp_loss`. Switched regression Ridge models to an explicit `lsqr` solver and classification linear models to explicit stable solvers. Added chunked classification probability/score prediction for base models and the classification meta-model to avoid native solver crashes observed in the `hackaton` environment.

Rationale: The scripts should fail with clear user-facing errors for invalid MLP losses, saved artifacts should be reloadable, and the default linear models should run reliably in the freshly created environment.

Affected files: `regression/__init__.py`, `classification/__init__.py`, `regression/models.py`, `classification/models.py`, `regression/train_superlearner_regression.py`, `classification/train_superlearner_classification.py`, `README.md`.

Verification: `python -m compileall -q regression classification` passed in `hackaton`. Regression smoke test passed with `ridge,knn` and `--use_conformal false`; regression artifacts loaded with `joblib`. Classification smoke tests passed with `ridge_classifier`, `logistic_regression`, and `logistic_regression,ridge_classifier` with conformal enabled; classification artifacts loaded with `joblib`. Invalid MLP losses fail before training with explicit allowed values.

Next step: For long hackathon runs, prefer starting with a small model list and then add optional XGBoost/LightGBM/CatBoost once the chosen columns and splits are final.
