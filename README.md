# Prediccion de Viralidad en YouTube
## Proyecto Final - Maquina de Aprendizaje 1
### Universidad de la Sabana

---

## Descripcion del problema

YouTube es la plataforma de video mas grande del mundo, con millones de videos publicados diariamente. Para creadores de contenido y marcas, predecir si un video tendra alto impacto de engagement (viralidad) es un problema estrategico de alto valor.

Este proyecto construye un sistema de clasificacion que predice si un video sera **viral** o **no viral**, a partir de sus caracteristicas contextuales y metricas de engagement relativo.

---

## Objetivo

Desarrollar un pipeline completo de aprendizaje de maquina bajo la metodologia **CRISP-DM** que:

1. Construya la variable objetivo `viralidad` a partir de views, likes, comentarios y shares.
2. Entrene y compare tres modelos: **Regresion Logistica**, **Random Forest** y **Gradient Boosting**.
3. Evalue el desempeno con **validacion cruzada estratificada** (5-fold) y conjunto de prueba separado.
4. Comunique los hallazgos a traves de un **dashboard** (Power BI / Tableau).

---

## Metodologia: CRISP-DM

| Fase                      | Descripcion                                                        |
|---------------------------|--------------------------------------------------------------------|
| 1. Comprension del negocio | Definicion del problema, objetivos analiticos y criterios de exito |
| 2. Comprension de los datos | EDA: distribuciones, correlaciones, calidad de datos               |
| 3. Preparacion de datos   | Limpieza, construccion de `viralidad`, ingenieria de caracteristicas |
| 4. Modelado               | Entrenamiento de 3 modelos con validacion cruzada                  |
| 5. Evaluacion             | Metricas, curvas ROC, matrices de confusion, importancia de features |
| 6. Conclusiones           | Hallazgos, limitaciones y recomendaciones                          |

---

## Variable objetivo: Viralidad

La variable `viralidad` se construye con el siguiente procedimiento:

1. **Transformacion log1p** de las 4 metricas de engagement para reducir el sesgo de distribucion de cola larga.
2. **Rango percentil** de cada metrica transformada (0 = minimo, 1 = maximo).
3. **Puntaje compuesto ponderado**:

```
engagement_score = 0.30 * rank_views + 0.25 * rank_likes + 0.20 * rank_comments + 0.25 * rank_shares
```

4. **Clasificacion binaria**: `viralidad = 1` si `engagement_score >= percentil 50`, `viralidad = 0` en caso contrario.

---

## Estructura del repositorio

```
Proyecto_Final_ML/
├── config/
│   └── config.yaml              # Parametros del proyecto
├── data/
│   ├── raw/                     # Dataset crudo (no versionado en git)
│   └── processed/
│       └── data_processed.csv   # Datos preprocesados para el dashboard
├── models/
│   ├── preprocessor.joblib      # ColumnTransformer ajustado
│   ├── regresion_logistica.joblib
│   ├── random_forest.joblib
│   └── gradient_boosting.joblib
├── notebooks/
│   └── 01_CRISP_DM_Viralidad_YouTube.ipynb  # Notebook principal CRISP-DM
├── reports/
│   └── figures/                 # Graficas generadas automaticamente
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Carga y perfilamiento de datos
│   ├── feature_engineering.py   # Construccion de viralidad y features
│   ├── preprocessing.py         # Pipeline de preprocesamiento
│   ├── modeling.py              # Entrenamiento y validacion cruzada
│   └── evaluation.py            # Metricas y visualizaciones
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Instrucciones de ejecucion

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd Proyecto_Final_ML
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Colocar el dataset crudo

Copiar `global_youtube_creator_data_large.csv` en la raiz del proyecto:

```
Proyecto_Final_ML/
└── global_youtube_creator_data_large.csv
```

### 4. Ejecutar el notebook

```bash
cd notebooks
jupyter notebook 01_CRISP_DM_Viralidad_YouTube.ipynb
```

Ejecutar todas las celdas en orden (Kernel > Restart & Run All).

---

## Resultados principales

| Modelo               | ROC-AUC (CV) | F1 (CV) | ROC-AUC (Test) | F1 (Test) |
|----------------------|:------------:|:-------:|:--------------:|:---------:|
| Regresion Logistica  | ~0.66        | ~0.62   | ~0.66          | ~0.61     |
| Random Forest        | ~0.68        | ~0.66   | ~0.68          | ~0.66     |
| Gradient Boosting    | ~0.68        | ~0.65   | ~0.68          | ~0.65     |

*Valores aproximados; los exactos se obtienen al ejecutar el notebook.*

**Mejor modelo:** Gradient Boosting (mayor ROC-AUC y F1)

### Caracteristicas mas importantes (modelos de arboles)
1. `engagement_rate` (tasa total de engagement)
2. `likes_per_view` (tasa de likes por vista)
3. `shares_per_view` (tasa de shares por vista)
4. `sentiment_score` (sentimiento de comentarios)
5. `duration_sec` (duracion del video)

---

## Dashboard

Los datos procesados (`data/processed/data_processed.csv`) junto con las figuras en `reports/figures/` alimentan el dashboard construido en **Power BI / Tableau**, que incluye:

- Distribucion de la variable `viralidad` y el `engagement_score`
- Engagement promedio por categoria, idioma y region
- Comparacion de metricas de los tres modelos
- Curvas ROC interactivas
- Mapa de importancia de caracteristicas

---

## Equipo

| Nombre | Universidad |
|--------|-------------|
| Martin | Universidad de la Sabana |

---

## Tecnologias

- **Python 3.10+**
- **scikit-learn** — modelos de ML y validacion cruzada
- **pandas / numpy** — manipulacion y analisis de datos
- **matplotlib / seaborn** — visualizaciones
- **joblib** — persistencia de modelos
- **Power BI / Tableau** — dashboard interactivo
