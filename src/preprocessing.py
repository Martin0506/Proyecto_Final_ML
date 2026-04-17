import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import os


def build_preprocessor(categorical_cols: list, numerical_cols: list,
                        binary_cols: list) -> ColumnTransformer:
    """
    Construye el ColumnTransformer que aplica:
    - OneHotEncoding a las columnas categoricas.
    - StandardScaler a las columnas numericas y binarias.

    Se elige OneHotEncoding (en lugar de LabelEncoding) para las
    categoricas porque los modelos lineales y de ensamble tratan
    cada categoria como una dimension independiente, evitando
    imponer un orden artificial entre clases.

    Parameters
    ----------
    categorical_cols : list  Columnas categoricas (category, language, region)
    numerical_cols   : list  Columnas numericas continuas
    binary_cols      : list  Columnas binarias (ads_enabled)

    Returns
    -------
    ColumnTransformer listo para fit/transform
    """
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numerical_transformer   = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numerical_transformer,   numerical_cols + binary_cols),
        ],
        remainder="drop"
    )
    return preprocessor


def split_data(df: pd.DataFrame, feature_cols: list, target_col: str,
               test_size: float = 0.20, random_state: int = 42):
    """
    Divide el dataset en conjuntos de entrenamiento y prueba con
    estratificacion por la variable objetivo para mantener la
    proporcion de clases en ambos subconjuntos.

    Parameters
    ----------
    df           : DataFrame procesado
    feature_cols : lista de columnas de entrada
    target_col   : nombre de la columna objetivo
    test_size    : fraccion para el conjunto de prueba
    random_state : semilla de aleatoriedad

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays numpy
    """
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print(f"[INFO] Entrenamiento: {X_train.shape[0]:,} muestras | Prueba: {X_test.shape[0]:,} muestras")
    print(f"[INFO] Balance en train - Viral: {y_train.mean()*100:.1f}% | No viral: {(1-y_train.mean())*100:.1f}%")
    print(f"[INFO] Balance en test  - Viral: {y_test.mean()*100:.1f}% | No viral: {(1-y_test.mean())*100:.1f}%")

    return X_train, X_test, y_train, y_test


def save_preprocessor(preprocessor, path: str = "models/preprocessor.joblib"):
    """Persiste el preprocesador ajustado en disco."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"[INFO] Preprocesador guardado en: {path}")


def load_preprocessor(path: str = "models/preprocessor.joblib"):
    """Carga un preprocesador previamente guardado."""
    return joblib.load(path)
