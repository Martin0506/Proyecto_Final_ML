import pandas as pd
import numpy as np

def build_viralidad(df: pd.DataFrame,
                    weights: dict = None,
                    threshold_pct: float = 50.0) -> pd.DataFrame:
    """
    Construye la variable binaria 'viralidad' a partir de views, likes,
    comments y shares mediante un puntaje de engagement compuesto.

    Metodologia:
    1. Aplicar transformacion log1p a cada metrica para reducir el sesgo
       provocado por valores extremos (distribucion tipo ley de potencia).
    2. Calcular el rango percentil de cada metrica transformada (0-1).
    3. Calcular un puntaje compuesto ponderado con los pesos configurados.
    4. Asignar viralidad = 1 si el puntaje supera el umbral percentil
       indicado (por defecto el 50 %, corte en la mediana).

    La logica de usar RANGOS PERCENTILES en lugar de los valores brutos
    garantiza que la variable refleja la posicion RELATIVA de cada video
    dentro de la distribucion, no su magnitud absoluta.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con columnas views, likes, comments, shares.
    weights : dict
        Pesos para cada metrica. Deben sumar 1.
        Por defecto: views=0.30, likes=0.25, comments=0.20, shares=0.25
    threshold_pct : float
        Percentil de corte para la clasificacion viral (default 50).

    Returns
    -------
    pd.DataFrame con las columnas adicionales:
        - log_views, log_likes, log_comments, log_shares
        - rank_views, rank_likes, rank_comments, rank_shares
        - engagement_score
        - viralidad (0/1)
    """
    df = df.copy()

    if weights is None:
        weights = {"views": 0.30, "likes": 0.25, "comments": 0.20, "shares": 0.25}

    for col in ["views", "likes", "comments", "shares"]:
        df[f"log_{col}"] = np.log1p(df[col])

    for col in ["views", "likes", "comments", "shares"]:
        df[f"rank_{col}"] = df[f"log_{col}"].rank(pct=True)

    df["engagement_score"] = (
        weights["views"]    * df["rank_views"] +
        weights["likes"]    * df["rank_likes"] +
        weights["comments"] * df["rank_comments"] +
        weights["shares"]   * df["rank_shares"]
    )

    umbral = np.percentile(df["engagement_score"], threshold_pct)
    df["viralidad"] = (df["engagement_score"] >= umbral).astype(int)

    n_viral = df["viralidad"].sum()
    print(f"[INFO] Umbral engagement_score (p{threshold_pct:.0f}): {umbral:.4f}")
    print(f"[INFO] Videos virales  : {n_viral:,} ({n_viral/len(df)*100:.1f}%)")
    print(f"[INFO] Videos no virales: {len(df)-n_viral:,} ({(len(df)-n_viral)/len(df)*100:.1f}%)")

    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera las caracteristicas que usaran los modelos para predecir viralidad.

    Decision de diseno:
    Las tasas de engagement (likes/views, comments/views, shares/views) son
    DISTINTAS de los valores absolutos usados para construir 'viralidad'.
    Un video puede tener muchas vistas pero poca interaccion relativa (o
    viceversa). Esto crea un problema de clasificacion genuinamente interesante:
    predecir viralidad a partir de la EFICIENCIA de engagement y el contexto
    del video, no de sus numeros absolutos.

    Caracteristicas generadas:
    - Tasas de engagement   : likes/views, comments/views, shares/views
    - Engagement total      : (likes+comments+shares) / (views+1)
    - Temporales            : hour, day_of_week, month
    - Categoricas           : category, language, region (ya en el dataset)
    - Numericas directas    : duration_sec, sentiment_score, ads_enabled

    Parameters
    ----------
    df : pd.DataFrame
        Dataset con columnas de metricas de engagement y timestamp.

    Returns
    -------
    pd.DataFrame con columnas de caracteristicas adicionales.
    """
    df = df.copy()


    views_safe = df["views"].clip(lower=1)

    df["likes_per_view"]    = df["likes"]    / views_safe
    df["comments_per_view"] = df["comments"] / views_safe
    df["shares_per_view"]   = df["shares"]   / views_safe
    df["engagement_rate"]   = (df["likes"] + df["comments"] + df["shares"]) / views_safe

    df["log_likes_per_view"]    = np.log1p(df["likes_per_view"])
    df["log_comments_per_view"] = np.log1p(df["comments_per_view"])
    df["log_shares_per_view"]   = np.log1p(df["shares_per_view"])
    df["log_engagement_rate"]   = np.log1p(df["engagement_rate"])

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"]       = df["timestamp"].dt.month

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["sent_x_engagement"]  = df["sentiment_score"] * df["log_engagement_rate"]
    df["dur_x_log_engagement"] = df["duration_sec"]  * df["log_engagement_rate"]

    df["ads_enabled"] = df["ads_enabled"].astype(int)

    return df

def get_feature_columns() -> dict:
    """
    Retorna el diccionario de columnas de caracteristicas por tipo,
    alineado con config.yaml.
    """
    return {
        "categorical": ["category", "language", "region"],
        "numerical": [
            "duration_sec", "sentiment_score",
            "likes_per_view", "comments_per_view", "shares_per_view", "engagement_rate",
            "log_likes_per_view", "log_comments_per_view",
            "log_shares_per_view", "log_engagement_rate",
            "sent_x_engagement", "dur_x_log_engagement",
            "hour", "day_of_week", "month",
        ],
        "binary": ["ads_enabled", "is_weekend"],
    }
