import pandas as pd
import numpy as np
import yaml

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(raw_path: str, sample_size: int = None,
                  random_state: int = 42) -> pd.DataFrame:
    """
    Carga el dataset crudo y opcionalmente extrae una muestra aleatoria
    estratificada para que sea representativa de todas las categorias.

    Parameters
    ----------
    raw_path : str
        Ruta al archivo CSV.
    sample_size : int, opcional
        Numero de registros a muestrear. None carga todo el dataset.
    random_state : int
        Semilla para reproducibilidad.

    Returns
    -------
    pd.DataFrame
    """
    print(f"[INFO] Cargando datos desde: {raw_path}")

    if sample_size:
        df = pd.read_csv(raw_path, low_memory=False)
        print(f"[INFO] Dataset completo: {df.shape[0]:,} filas x {df.shape[1]} columnas")
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        print(f"[INFO] Muestra extraida: {df.shape[0]:,} filas")
    else:
        df = pd.read_csv(raw_path, low_memory=False)
        print(f"[INFO] Dataset completo cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")

    return df


def basic_profiling(df: pd.DataFrame) -> None:
    """Imprime un reporte basico del dataset."""
    print("\n" + "="*60)
    print("PERFIL BASICO DEL DATASET")
    print("="*60)
    print(f"\nDimensiones  : {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print(f"\nTipos de dato:\n{df.dtypes}")
    print(f"\nValores nulos:\n{df.isnull().sum()}")
    print(f"\nDuplicados   : {df.duplicated().sum():,}")
    print(f"\nEstadisticas descriptivas:\n{df.describe(include='all').T}")
