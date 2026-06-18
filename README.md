Machine Learning Practices 

Este repositorio contiene el compendio completo de actividades prácticas, desarrollos matemáticos y programas en Python desarrollados para la materia **Temas Selectos II: Machine Learning** en la **Universidad Autónoma Metropolitana**.

El proyecto abarca desde la deducción analítica de algoritmos clásicos hasta su implementación en código, utilizando librerías científicas y visualizaciones avanzadas para evaluar su rendimiento.

## Estructura del repo
El repositorio está organizado de la siguiente manera para facilitar la ejecución de los programas y la compilación del reporte:

```text
├── data/                         # Bases de datos utilizadas por los scripts
│   ├── precios_casas.csv         # Dataset para Regresión Lineal y RANSAC
│   └── (otros_datasets...)       # Datasets sintéticos o dummies
│
├── src/                          # Código fuente de los programas (.py o .ipynb)
│   ├── 01_gradiente_descendiente.py
│   ├── 02_regresion_ransac.py
│   ├── 03_regresion_multilineal.py
│   ├── 04_clasificacion_softmax.py
│   ├── 05_naive_bayes.py
│   ├── 06_svm_lineal_y_xor.py
│   ├── 07_knn_iris.py
│   └── 08_pca_knn_iris.py
│
├── img/                          # Gráficos generados por los programas (para el PDF)
│   ├── uam-azcapotzalco.png
│   ├── graddesc.png
│   ├── corr.png
│   ├── Ransac.png
│   ├── confusionmatrix.png
│   └── (demás_figuras...)
│
├── main.tex                      # Documento fuente en LaTeX corregido
├── ML_TareasCompletas.pdf        # Reporte final compilado
└── README.md                     # Este archivo

```

## Contenido y Correspondencia de Actividades

A continuación se detalla qué cubre cada sección del reporte y qué script de la carpeta `src/` se encarga de su simulación:

1. **Deducciones Analíticas (Actividades 1, 2 y 8):** Desarrollos matemáticos en $\LaTeX$ sobre ecuaciones normales, soluciones cerradas matriciales ($\boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$) y la derivada de la Entropía Cruzada con Softmax mediante regla de la cadena.
2. **Gradiente Descendiente (Actividad 3):** Simulación e instrumentación del camino de optimización. Visualización tridimensional de la superficie de costo y curvas de contorno. `[src/01_gradiente_descendiente.py]`
3. **Regresión Lineal vs. RANSAC (Actividad 4 y 5):** Análisis de correlación e impacto de valores atípicos (*outliers*) sobre el dataset de viviendas, evaluado con $R^2$ y estadístico F. `[src/02_regresion_ransac.py]`
4. **Regresión Multilineal (Actividad 6):** Ajuste de hiperplanos utilizando múltiples variables independientes para la predicción de precios. `[src/03_regresion_multilineal.py]`
5. **Clasificación Softmax (Actividad 7):** Clasificador multiclase evaluado a través de matrices de confusión (*heatmaps*) y análisis de curvas ROC / AUC. `[src/04_clasificacion_softmax.py]`
6. **Naive Bayes (Actividad 9):** Clasificación probabilística gaussiana aplicada al conjunto de datos *Iris*. `[src/05_naive_bayes.py]`
7. **Support Vector Machines - SVM (Actividad 10 y 11):** Margen máximo para datos linealmente separables (Dummy) y uso del *kernel trick* (RBF) para resolver el problema no lineal XOR. `[src/06_svm_lineal_y_xor.py]`
8. **KNN & Reducción de Dimensionalidad PCA (Actividad 12, 13 y más):** Implementación de K-Nearest Neighbors y optimización del espacio de características mediante Análisis de Componentes Principales por el método de la matriz de covarianza. `[src/07_knn_iris.py]` y `[src/08_pca_knn_iris.py]`

## Instalación y Uso

### 1. Clonar el repositorio

```bash
git clone [https://github.com/TU_USUARIO/TU_REPOSITORIO.git](https://github.com/TU_USUARIO/TU_REPOSITORIO.git)
cd TU_REPOSITORIO

```

### 2. Instalar dependencias

Se recomienda utilizar un entorno virtual. Las librerías principales de Python requeridas son `numpy`, `scikit-learn`, `matplotlib`, `seaborn` y `pandas`.

```bash
pip install -r requirements.txt

```

*(Nota: Recuerda añadir un archivo `requirements.txt` en la raíz si usas este comando, o simplemente listar las librerías aquí).*

### 3. Ejecución de los programas

Los scripts están diseñados para leer automáticamente los archivos dentro de la carpeta `data/` y guardar los gráficos resultantes directamente en la carpeta `img/`. Por ejemplo:

```bash
python src/02_regresion_ransac.py

```

## Autores

* **Alumno:** Braulio Leonardo Santa Fe García – *Departamento de Sistemas, UAM Azcapotzalco*
* **Profesor:** Arturo Zúñiga López

```


```

De esta forma, no tendrás que cambiar la ruta de cada `\includegraphics` uno por uno; $\LaTeX$ sabrá automáticamente que debe buscar todas las imágenes dentro de la carpeta `img/`.
