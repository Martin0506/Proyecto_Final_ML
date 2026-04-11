"""
evaluation.py
-------------
Funciones de evaluacion, metricas y visualizacion de resultados.

Fase CRISP-DM: Evaluacion
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)


# Paleta consistente para los 3 modelos
MODEL_COLORS = {
    "Regresion Logistica": "#4C72B0",
    "Random Forest"      : "#55A868",
    "Gradient Boosting"  : "#C44E52",
}


def evaluate_on_test(models: dict, X_test: np.ndarray,
                     y_test: np.ndarray) -> pd.DataFrame:
    """
    Evalua cada modelo sobre el conjunto de prueba y retorna un DataFrame
    con las metricas de clasificacion.

    Parameters
    ----------
    models  : dict {nombre: modelo entrenado}
    X_test  : features de prueba preprocesadas
    y_test  : etiquetas reales

    Returns
    -------
    pd.DataFrame con metricas por modelo.
    """
    records = []
    for name, model in models.items():
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        records.append({
            "Modelo"    : name,
            "Accuracy"  : accuracy_score(y_test, y_pred),
            "F1"        : f1_score(y_test, y_pred),
            "ROC-AUC"   : roc_auc_score(y_test, y_proba),
            "Precision" : precision_score(y_test, y_pred),
            "Recall"    : recall_score(y_test, y_pred),
        })

        print(f"\n{'='*55}")
        print(f"  {name}")
        print(f"{'='*55}")
        print(classification_report(y_test, y_pred,
                                    target_names=["No Viral", "Viral"]))

    return pd.DataFrame(records).set_index("Modelo")


def plot_confusion_matrices(models: dict, X_test: np.ndarray,
                            y_test: np.ndarray,
                            save_path: str = "reports/figures/confusion_matrices.png"):
    """
    Genera y guarda las matrices de confusion para los tres modelos
    en una sola figura de 1x3.
    """
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    fig.suptitle("Matrices de Confusion - Conjunto de Prueba", fontsize=14, fontweight="bold")

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=["No Viral", "Viral"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Figura guardada: {save_path}")


def plot_roc_curves(models: dict, X_test: np.ndarray, y_test: np.ndarray,
                    save_path: str = "reports/figures/roc_curves.png"):
    """
    Genera y guarda las curvas ROC de los tres modelos superpuestas
    para facilitar la comparacion visual.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        color = MODEL_COLORS.get(name, None)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Clasificador aleatorio")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=11)
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=11)
    ax.set_title("Curvas ROC - Comparacion de Modelos", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Figura guardada: {save_path}")


def plot_feature_importance(model, feature_names: list, model_name: str,
                             top_n: int = 15,
                             save_path: str = "reports/figures/feature_importance.png"):
    """
    Grafica la importancia de caracteristicas para modelos basados en
    arboles (RandomForest y GradientBoosting).  Para Regresion Logistica
    grafica los coeficientes absolutos.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title_type = "Importancia de Caracteristicas (Gini)"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        title_type = "Coeficientes Absolutos"
    else:
        print(f"[WARN] El modelo {model_name} no tiene importancias ni coeficientes.")
        return

    indices = np.argsort(importances)[::-1][:top_n]
    feat_names = [feature_names[i] for i in indices]
    feat_vals  = importances[indices]
    color = MODEL_COLORS.get(model_name, "#888888")

    ax.barh(range(top_n), feat_vals[::-1], color=color, alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(feat_names[::-1])
    ax.set_xlabel(title_type, fontsize=11)
    ax.set_title(f"Top {top_n} Caracteristicas - {model_name}", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    final_path = save_path.replace(".png", f"_{safe_name}.png")
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    plt.savefig(final_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Figura guardada: {final_path}")


def plot_cv_comparison(cv_results: pd.DataFrame,
                        save_path: str = "reports/figures/cv_comparison.png"):
    """
    Grafica de barras comparando ROC-AUC y F1 de la validacion cruzada
    con barras de error que muestran la desviacion estandar.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    metrics = [("ROC-AUC (val)", "ROC-AUC std", "ROC-AUC"),
               ("F1 (val)",      "F1 std",       "F1-Score")]

    colors = [MODEL_COLORS.get(m, "#888") for m in cv_results.index]

    for ax, (metric, std_col, title) in zip(axes, metrics):
        vals = cv_results[metric].values
        stds = cv_results[std_col].values
        bars = ax.bar(cv_results.index, vals, yerr=stds, capsize=5,
                      color=colors, alpha=0.85, error_kw={"linewidth": 1.5})
        ax.set_title(f"{title} - Validacion Cruzada (5-fold)", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", alpha=0.3)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Comparacion de Modelos - Validacion Cruzada Estratificada",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Figura guardada: {save_path}")


def plot_metrics_heatmap(test_results: pd.DataFrame,
                          save_path: str = "reports/figures/metrics_heatmap.png"):
    """
    Mapa de calor con las metricas finales de los tres modelos en el
    conjunto de prueba para una lectura rapida y comparativa.
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.heatmap(test_results, annot=True, fmt=".4f", cmap="YlGnBu",
                linewidths=0.5, ax=ax, vmin=0, vmax=1)
    ax.set_title("Metricas en Conjunto de Prueba - Comparacion Final",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Figura guardada: {save_path}")
