# SuperLearner with Conformal Prediction

Proyecto base en Python para entrenar un SuperLearner sobre datos tabulares, con una capa de prediccion conformal para cuantificar incertidumbre. Incluye pipelines separados para regresion y clasificacion, validacion cruzada, preprocesamiento automatico de columnas numericas y categoricas, logging y guardado de artefactos.

## Estructura

```text
regression/
  train_superlearner_regression.py
  conformal_regression.py
  models.py
  utils.py
  progress.py
classification/
  train_superlearner_classification.py
  conformal_classification.py
  models.py
  utils.py
  progress.py
README.md
requirements.txt
```

## Instalacion

```bash
pip install -r requirements.txt
```

Las librerias `xgboost`, `lightgbm` y `catboost` se validan en tiempo de ejecucion. Si seleccionas uno de esos modelos y falta el paquete, el script mostrara un error con este comando:

```bash
pip install xgboost lightgbm catboost
```

## Ejecucion para regresion

```bash
python regression/train_superlearner_regression.py \
  --data_path data/regression_data.csv \
  --target_column y \
  --feature_columns all \
  --continuous_columns all \
  --cv_folds 5 \
  --models all \
  --mlp_loss squared_error \
  --output_dir results/regression \
  --alpha 0.1 \
  --use_conformal true \
  --random_state 42
```

## Ejecucion para clasificacion

```bash
python classification/train_superlearner_classification.py \
  --data_path data/classification_data.csv \
  --target_column label \
  --feature_columns all \
  --ohe_columns all \
  --cv_folds 5 \
  --models all \
  --mlp_loss log_loss \
  --output_dir results/classification \
  --alpha 0.1 \
  --use_conformal true \
  --random_state 42
```

## Ejemplos completos con todas las flags

Hay dos modos de entrada excluyentes:

```text
Modo 1: --data_path
El script carga un unico CSV y crea internamente train, validacion y conformal.

Modo 2: --train_data_path + --validation_data_path + opcional/obligatorio --conformal_data_path
Usa CSVs ya preprocesados y separados previamente.
```

Si `--use_conformal true`, hace falta un conjunto conformal. En modo `--data_path` se crea automaticamente; en modo de CSVs separados debes pasar `--conformal_data_path`. Si `--use_conformal false`, no se calibra prediccion conformal y `--conformal_data_path` no es necesario.

### Clasificacion con un unico CSV

```bash
python classification/train_superlearner_classification.py \
  --data_path data.csv \
  --target_column gutper \
  --feature_columns ccl,icl \
  --continuous_columns none \
  --ohe_columns ccl,icl \
  --ordinal_columns none \
  --cv_folds 3 \
  --models logistic_regression,knn,naive_bayes \
  --mlp_loss log_loss \
  --output_dir results/classification_pilot_selected_columns \
  --alpha 0.1 \
  --use_conformal true \
  --random_state 42
```

### Clasificacion con CSVs ya separados

```bash
python classification/train_superlearner_classification.py \
  --train_data_path data/splits/classification_train.csv \
  --validation_data_path data/splits/classification_validation.csv \
  --conformal_data_path data/splits/classification_conformal.csv \
  --target_column gutper \
  --feature_columns ccl,icl \
  --continuous_columns none \
  --ohe_columns ccl,icl \
  --ordinal_columns none \
  --cv_folds 5 \
  --models logistic_regression,random_forest,gradient_boosting \
  --mlp_loss log_loss \
  --output_dir results/classification_presplit \
  --alpha 0.1 \
  --use_conformal true \
  --random_state 42
```

### Regresion con CSVs ya separados y sin conformal

```bash
python regression/train_superlearner_regression.py \
  --train_data_path data/splits/regression_train.csv \
  --validation_data_path data/splits/regression_validation.csv \
  --target_column y \
  --feature_columns age,bmi,sex,center_id \
  --continuous_columns age,bmi \
  --ohe_columns sex \
  --ordinal_columns center_id \
  --cv_folds 5 \
  --models ridge,random_forest,gradient_boosting,mlp \
  --mlp_loss squared_error \
  --output_dir results/regression_presplit_no_conformal \
  --alpha 0.1 \
  --use_conformal false \
  --random_state 42
```

## Referencia de flags

Las flags son las mismas para regresion y clasificacion.

| Flag | Obligatoria | Descripcion |
| --- | --- | --- |
| `--data_path` | Si no usas splits externos | CSV unico de entrada. El script crea internamente train, validacion y, si aplica, conformal. |
| `--train_data_path` | Si usas splits externos | CSV de entrenamiento ya separado. No se puede combinar con `--data_path`. |
| `--validation_data_path` | Si usas splits externos | CSV de validacion/evaluacion final ya separado. No se puede combinar con `--data_path`. |
| `--conformal_data_path` | Si usas splits externos y `--use_conformal true` | CSV de calibracion conformal ya separado. |
| `--target_column` | Si | Nombre de la columna objetivo. |
| `--feature_columns` | No, default `all` | Columnas predictoras. Usa `all` para todas menos el target, o una lista como `ccl,icl`. |
| `--continuous_columns` | No | Subconjunto de features a tratar como continuas: conversion numerica, imputacion mediana y escalado. Acepta `none`, `all` o lista. |
| `--ohe_columns` | No | Subconjunto de features discretas/categoricas a codificar con one-hot encoding. Acepta `none`, `all` o lista. |
| `--ordinal_columns` | No | Subconjunto de features discretas que no quieres one-hot encodear. Usa `OrdinalEncoder`. Acepta `none`, `all` o lista. |
| `--cv_folds` | Si | Numero de folds para generar predicciones out-of-fold del SuperLearner. |
| `--models` | Si | Modelos base. Usa `all` o una lista separada por comas. |
| `--mlp_loss` | No | Loss para el modelo `mlp`. En clasificacion usa `log_loss`. En regresion usa `squared_error` o `poisson` si tu version de scikit-learn lo soporta. |
| `--output_dir` | Si | Carpeta donde se guardan modelos, metricas, predicciones, configuracion y logs. |
| `--alpha` | No, default `0.1` | Nivel de error conformal. `0.1` apunta a cobertura aproximada del 90%. |
| `--use_conformal` | No, default `true` | Activa o desactiva prediccion conformal. Acepta `true/false`, `yes/no` o `1/0`. |
| `--random_state` | No, default `42` | Semilla para splits y modelos reproducibles. |

Reglas importantes:

- `--data_path` y los flags `--train_data_path`, `--validation_data_path`, `--conformal_data_path` son modos excluyentes.
- Si usas splits externos y `--use_conformal true`, `--conformal_data_path` es obligatorio.
- Si usas splits externos y `--use_conformal false`, basta con `--train_data_path` y `--validation_data_path`.
- Las columnas de `--continuous_columns`, `--ohe_columns` y `--ordinal_columns` deben estar dentro de `--feature_columns`.
- Una columna no puede aparecer en mas de uno de esos tres grupos de tipado.
- Si no indicas tipado, el script autodetecta: numericas como continuas y no numericas como one-hot.

## Seleccion de modelos

Puedes usar todos los modelos disponibles:

```bash
--models all
```

O una lista separada por comas:

```bash
--models random_forest,extra_trees,gradient_boosting,svm,knn
```

Para usar una red neuronal MLP:

```bash
--models mlp
```

La loss de MLP se controla con:

```bash
--mlp_loss log_loss
```

En clasificacion, usa `log_loss`. En regresion, usa `squared_error` o `poisson` si la version instalada de scikit-learn lo soporta. Si pides una loss no valida, el script falla antes de entrenar.

## Seleccion de columnas

Por defecto se usan todas las columnas excepto la columna objetivo:

```bash
--feature_columns all
```

Tambien puedes indicar una lista separada por comas:

```bash
--feature_columns ccl,icl
```

La columna objetivo no puede aparecer en `--feature_columns`, para evitar leakage.

## Tipado de columnas

El preprocesamiento se puede controlar con tres flags:

```text
--continuous_columns: columnas continuas. Se convierten a numerico, se imputan con mediana y se escalan.
--ohe_columns: columnas discretas/categoricas con one-hot encoding.
--ordinal_columns: columnas discretas que no quieres one-hot encodear; se imputan y se codifican con OrdinalEncoder.
```

Valores utiles:

```text
none: no asigna columnas a ese grupo.
all: asigna todas las columnas seleccionadas con --feature_columns a ese grupo.
lista separada por comas: asigna solo esas columnas.
```

Si no especificas ninguna de estas flags, el script autodetecta: columnas numericas como continuas y el resto como one-hot. Las columnas no pueden estar en mas de un grupo. Todas las columnas indicadas en estas flags deben estar incluidas en `--feature_columns`.

Ejemplo:

```bash
--feature_columns age,bmi,sex,center_id \
--continuous_columns age,bmi \
--ohe_columns sex \
--ordinal_columns center_id
```

En el CSV piloto, `ccl` e `icl` son categoricas razonables para one-hot:

```bash
--feature_columns ccl,icl \
--continuous_columns none \
--ohe_columns ccl,icl \
--ordinal_columns none
```

`hmdb_id`, `name` e `inchi` tienen alta cardinalidad. Si se incluyen como one-hot, el modelo guardado puede crecer mucho y el entrenamiento puede memorizar identificadores.

## Prediccion conformal

La prediccion conformal se controla con:

```bash
--use_conformal true
--alpha 0.1
```

`--alpha` es el nivel de error esperado. Con `0.1`, el objetivo conformal aproximado es cobertura del 90%.

Para desactivar conformal:

```bash
--use_conformal false
```

En ese caso se entrenan y evaluan los modelos, pero `q_hat`, `conformal_coverage`, `mean_interval_width` o `mean_prediction_set_size` quedan como `null`, y las columnas conformales de `predictions.csv` se guardan vacias o como `NaN`.

Ejemplo con el CSV piloto incluido:

```bash
python classification/train_superlearner_classification.py \
  --data_path data.csv \
  --target_column gutper \
  --feature_columns ccl,icl \
  --cv_folds 3 \
  --models logistic_regression,knn,naive_bayes \
  --output_dir results/classification_pilot_selected_columns
```

### Ejemplo solo con XGBoost, LightGBM y CatBoost en regresion

```bash
python regression/train_superlearner_regression.py \
  --data_path data/regression_data.csv \
  --target_column y \
  --feature_columns all \
  --cv_folds 5 \
  --models xgboost_regressor,lightgbm_regressor,catboost_regressor \
  --output_dir results/regression
```

### Ejemplo solo con XGBoost, LightGBM y CatBoost en clasificacion

```bash
python classification/train_superlearner_classification.py \
  --data_path data/classification_data.csv \
  --target_column label \
  --feature_columns all \
  --cv_folds 5 \
  --models xgboost_classifier,lightgbm_classifier,catboost_classifier \
  --output_dir results/classification
```

## Modelos disponibles

Regresion:

```text
linear_model, ridge, lasso, elastic_net, random_forest, extra_trees,
gradient_boosting, hist_gradient_boosting, svm, knn, mlp,
xgboost_regressor, lightgbm_regressor, catboost_regressor
```

Clasificacion:

```text
logistic_regression, ridge_classifier, random_forest, extra_trees,
gradient_boosting, hist_gradient_boosting, svm, knn, naive_bayes, mlp,
xgboost_classifier, lightgbm_classifier, catboost_classifier
```

## Que es un SuperLearner

Un SuperLearner es un ensamble apilado. Primero entrena varios modelos base mediante validacion cruzada y genera predicciones out-of-fold. Esas predicciones se usan como nuevas variables para entrenar un meta-modelo. En inferencia, los modelos base producen predicciones y el meta-modelo aprende como combinarlas.

## Que es la prediccion conformal

La prediccion conformal usa un conjunto de calibracion separado para estimar la incertidumbre de las predicciones. En regresion, calcula residuos absolutos y construye intervalos alrededor de la prediccion. En clasificacion, usa la probabilidad asignada a la clase verdadera para crear conjuntos de clases plausibles. La flag `--alpha` controla la tasa de error esperada; por defecto es `0.1`.

## Archivos generados

Cada ejecucion guarda en `output_dir`:

```text
trained_superlearner.joblib
base_models.joblib
metrics.json
predictions.csv
config.json
training_log.txt
```

En regresion, `predictions.csv` incluye:

```text
y_true, y_pred, lower_bound, upper_bound
```

En clasificacion, `predictions.csv` incluye:

```text
y_true, y_pred, predicted_probability, conformal_prediction_set
```

`metrics.json` contiene metricas finales. Para regresion incluye `MAE`, `RMSE`, `R2`, `conformal_coverage` y `mean_interval_width`. Para clasificacion incluye `accuracy`, `balanced_accuracy`, `f1_macro`, `log_loss`, `conformal_coverage` y `mean_prediction_set_size`.
