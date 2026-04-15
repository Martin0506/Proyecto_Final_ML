"""
modeling.py
-----------
Definicion, ajuste de hiperparametros y validacion cruzada de los tres modelos:
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
from sklearn.model_selection import cross_validate, StratifiedKFold, RandomizedSearchCV
import scipy.stats as stats


# ---------------------------------------------------------------------------
# DEFINICION DE MODELOS BASE
# ---------------------------------------------------------------------------

def get_base_models(random_state: int = 42) -> dict:
    """
    Retorna un diccionario con los clasificadores base instanciados.
    """
    return {
        "Regresion Logistica": LogisticRegression(
            solver="lbfgs",
            max_iter=2000, # Aumentado para asegurar convergencia en tuning
            random_state=random_state,
            n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=random_state,
        ),
    }


# ---------------------------------------------------------------------------
# AJUSTE DE HIPERPARAMETROS (TUNING)
# ---------------------------------------------------------------------------

def get_param_distributions() -> dict:
    """
    Define los espacios de busqueda para RandomizedSearchCV.
    """
    return {
        "Regresion Logistica": {
            "C": stats.loguniform(1e-3, 1e2),
            "class_weight": [None, "balanced"]
        },
        "Random Forest": {
            "n_estimators": stats.randint(100, 300),
            "max_depth": [None, 10, 15, 20, 25],
            "min_samples_split": stats.randint(2, 15),
            "min_samples_leaf": stats.randint(1, 10),
            "class_weight": [None, "balanced"]
        },
        "Gradient Boosting": {
            "n_estimators": stats.randint(100, 300),
            "learning_rate": stats.loguniform(0.01, 0.2),
            "max_depth": stats.randint(3, 8),
            "subsample": stats.uniform(0.7, 0.3) # 0.7 a 1.0
        }
    }


def tune_hyperparameters(models: dict, X_train: np.ndarray, y_train: np.ndarray,
                         n_iter: int = 15, cv_folds: int = 3, random_state: int = 42) -> dict:
    """
    Ejecuta RandomizedSearchCV para encontrar los mejores hiperparametros.
    
    Se usan 3 folds en lugar de 5 durante el tuning para ahorrar tiempo computacional 
    en un dataset de 100k filas, optimizando sobre ROC-AUC.
    """
    param_grids = get_param_distributions()
    best_models = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    for name, model in models.items():
        print(f"\n[TUNING] Iniciando busqueda para: {name}")
        t0 = time.time()
        
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grids[name],
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            random_state=random_state,
            verbose=1
        )
        
        search.fit(X_train, y_train)
        elapsed = time.time() - t0
        
        print(f"    Mejor ROC-AUC (CV): {search.best_score_:.4f}")
        print(f"    Mejores params: {search.best_params_}")
        print(f"    Tiempo de busqueda: {elapsed:.1f}s")
        
        best_models[name] = search.best_estimator_

    return best_models


# ---------------------------------------------------------------------------
# VALIDACION CRUZADA (EVALUACION DEL MEJOR MODELO)
# ---------------------------------------------------------------------------

def cross_validate_models(models: dict, X_train: np.ndarray,
                           y_train: np.ndarray, cv_folds: int = 5,
                           random_state: int = 42) -> pd.DataFrame:
    """
    Ejecuta validacion cruzada estratificada sobre los modelos YA OPTIMIZADOS
    para obtener un reporte robusto antes de pasar al conjunto de prueba.
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
        print(f"\n[CV] Validando modelo optimizado: {name} ({cv_folds} folds)...")
        t0 = time.time()

        # --- CORRECCIÓN AQUÍ: return_train_score=True ---
        scores = cross_validate(
            model, X_train, y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=True, 
            n_jobs=-1,
        )

        elapsed = time.time() - t0

        # --- CORRECCIÓN AQUÍ: Agregamos "Accuracy (train)" al diccionario ---
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

        print(f"    F1       : {results[name]['F1 (val)']:.4f} ± {results[name]['F1 std']:.4f}")
        print(f"    ROC-AUC  : {results[name]['ROC-AUC (val)']:.4f} ± {results[name]['ROC-AUC std']:.4f}")

    return pd.DataFrame(results).T


# ---------------------------------------------------------------------------
# ENTRENAMIENTO FINAL Y PERSISTENCIA
# ---------------------------------------------------------------------------

def train_and_save(model, model_name: str, X_train: np.ndarray,
                   y_train: np.ndarray, models_dir: str = "models") -> object:
    """
    Entrena el modelo final y lo persiste en disco.
    """
    os.makedirs(models_dir, exist_ok=True)
    filename = model_name.lower().replace(" ", "_") + ".joblib"
    filepath = os.path.join(models_dir, filename)

    print(f"[INFO] Entrenando modelo final {model_name}...")
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"[INFO] {model_name} entrenado en {elapsed:.1f}s -> guardado en {filepath}")

    joblib.dump(model, filepath)
    return model