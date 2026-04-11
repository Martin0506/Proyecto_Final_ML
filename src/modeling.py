"""
modeling.py
-----------
Definicion, entrenamiento y validacion cruzada de los tres modelos:
  1. Regresion Logistica
  2. Random Forest
  3. Gradient Boosting

Fase CRISP-DM: Modelado
"""

import numpy as np
import pandas as pd
import joblib
import os
import time

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold


# ---------------------------------------------------------------------------
# DEFINICION DE MODELOS
# ---------------------------------------------------------------------------

def get_models(random_state: int = 42) -> dict:
    """
    Retorna un diccionario con los tres clasificadores configurados.

    Justificacion de hiperparametros:
    - LogisticRegression : C=1.0 (regularizacion L2 por defecto), solver lbfgs
      apropiado para datasets multiclase medianos; max_iter=1000 asegura
      convergencia en datos escalados.
    - RandomForest : n_estimators=200 (buen balance bias-varianza), max_depth=12
      evita sobreajuste en 100k muestras, min_samples_leaf=10 reduce ruido.
    - GradientBoosting : learning_rate bajo (0.05) con mas arboles (200) es mas
      robusto que learning_rate alto con pocos arboles; subsample=0.8 agrega
      estocasticidad (similar a SGD) para regularizacion adicional.
    """
    return {
        "Regresion Logistica": LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=random_state,
        ),
    }


# ---------------------------------------------------------------------------
# VALIDACION CRUZADA
# ---------------------------------------------------------------------------

def cross_validate_models(models: dict, X_train: np.ndarray,
                           y_train: np.ndarray, cv_folds: int = 5,
                           random_state: int = 42) -> pd.DataFrame:
    """
    Ejecuta validacion cruzada estratificada (StratifiedKFold) para cada
    modelo y retorna un DataFrame con las metricas promedio.

    Se usa StratifiedKFold para garantizar que cada fold mantenga la
    proporcion de clases del conjunto original, lo cual es especialmente
    importante con datasets desbalanceados.

    Metricas reportadas:
    - Accuracy  : fraccion de predicciones correctas
    - F1        : media armonica de precision y recall (macro)
    - ROC-AUC   : area bajo la curva ROC; mide discriminacion entre clases
    - Precision : verdaderos positivos / (VP + FP)
    - Recall    : verdaderos positivos / (VP + FN)

    Parameters
    ----------
    models       : dict {nombre: clasificador}
    X_train      : array de caracteristicas preprocesadas (train)
    y_train      : array de etiquetas (train)
    cv_folds     : numero de folds (default 5)
    random_state : semilla

    Returns
    -------
    pd.DataFrame con una fila por modelo y columnas por metrica.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    scoring = {
        "accuracy"  : "accuracy",
        "f1"        : "f1",
        "roc_auc"   : "roc_auc",
        "precision" : "precision",
        "recall"    : "recall",
    }

    results = {}

    for name, model in models.items():
        print(f"\n[CV] Validando: {name} ({cv_folds} folds)...")
        t0 = time.time()

        scores = cross_validate(
            model, X_train, y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
        )

        elapsed = time.time() - t0

        results[name] = {
            "Accuracy (val)"    : scores["test_accuracy"].mean(),
            "Accuracy std"      : scores["test_accuracy"].std(),
            "F1 (val)"          : scores["test_f1"].mean(),
            "F1 std"            : scores["test_f1"].std(),
            "ROC-AUC (val)"     : scores["test_roc_auc"].mean(),
            "ROC-AUC std"       : scores["test_roc_auc"].std(),
            "Precision (val)"   : scores["test_precision"].mean(),
            "Recall (val)"      : scores["test_recall"].mean(),
            "Accuracy (train)"  : scores["train_accuracy"].mean(),
            "Tiempo (s)"        : round(elapsed, 1),
        }

        print(f"    Accuracy : {results[name]['Accuracy (val)']:.4f} ± {results[name]['Accuracy std']:.4f}")
        print(f"    F1       : {results[name]['F1 (val)']:.4f} ± {results[name]['F1 std']:.4f}")
        print(f"    ROC-AUC  : {results[name]['ROC-AUC (val)']:.4f} ± {results[name]['ROC-AUC std']:.4f}")
        print(f"    Tiempo   : {elapsed:.1f}s")

    return pd.DataFrame(results).T


# ---------------------------------------------------------------------------
# ENTRENAMIENTO FINAL Y PERSISTENCIA
# ---------------------------------------------------------------------------

def train_and_save(model, model_name: str, X_train: np.ndarray,
                   y_train: np.ndarray, models_dir: str = "models") -> object:
    """
    Entrena el modelo sobre el conjunto completo de entrenamiento y
    lo persiste en disco con joblib.

    Parameters
    ----------
    model      : clasificador sklearn
    model_name : nombre legible (usado para el nombre de archivo)
    X_train    : features de entrenamiento preprocesadas
    y_train    : etiquetas de entrenamiento
    models_dir : directorio de destino

    Returns
    -------
    Modelo entrenado.
    """
    os.makedirs(models_dir, exist_ok=True)
    filename = model_name.lower().replace(" ", "_") + ".joblib"
    filepath = os.path.join(models_dir, filename)

    print(f"[INFO] Entrenando {model_name}...")
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"[INFO] {model_name} entrenado en {elapsed:.1f}s -> guardado en {filepath}")

    joblib.dump(model, filepath)
    return model
