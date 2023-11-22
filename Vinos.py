#Introduccion:
#Los datos representan los estudios quimicos realizados en vinos provenientes de tres diferentes cultivos en Italia
#El programa estudia el contenido de 13 principales indicadores en los 3 tipos de Vino distintos
#La lista de indicadores es: 
# 1) Alcohol
# 2) Ácido málico
# 3) Ceniza
# 4) Alcalinidad de la ceniza
# 5) Magnesio
# 6) Fenoles totales
# 7) Flavonoides
# 8) Fenoles no flavonoides
# 9) Proantocianidinas
# 10) Intensidad del color
# 11) Tono
# 12) OD280/OD315 de vinos diluidos
# 13) Prolina   

#El numero de estudios para cada tipo se resume como sigue: 
#tipo 1: 59
#tipo 2: 71
#tipo 3: 48

#El proyecto primeramente se enfoca en comparar las tecnicas de clusterizacion de Kmeans para evaluarlas y 
#posteriormente cluseterizar los vinos de acuerdo a los indicadores mencionados ajustando a la tecnica más
#favorable


#Importamos las librerias y los datos con las funciones de scikit-learn mas conocida como sklearn
import numpy as np
from sklearn.datasets import load_wine
data, labels = load_wine(return_X_y=True)
(nMuestras, nIndicadores)= data.shape
nWine = np.unique(labels).size
print(f"\n# Tipos de Vinos: {nWine}; # Muestras: {nMuestras}; # Indicadores: {nIndicadores}")

#Realizamos nuestro criterio de evaluación
#Esta funcion evaluara los metodos de clusterizacion para K-means que ya esta desarrollado en scitkit-learn

from time import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

def Evalua_kmeans(kmeans, nombre, data, labels): #nombre de la estrategia 
    t0 = time()   #tiempo inicial
    estimador = make_pipeline(StandardScaler(), kmeans) #creamos nuestro estimador
    #Con make_pipeline creamos el pipeline sin necesidad de nombrar cada etapa
    #StandardScaler se utiliza para escalar los datos de tal forma que tengan media 0 y desvSt 1
    estimador=estimador.fit(data) #*******    Implementamos el método K-means  *****
    fit_time = time() - t0 #medimos el tiempo que tardo en realizar el método
    inertia = estimador[-1].inertia_ #accedemos a la suma de las distancias euclideanas al centroide de la ultima transformación
    results = [nombre, fit_time, inertia]
    
    #A continuacion se establecen los puntos a evaluar
    metricas = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    
    #extraemos las etiquetas de las metricas
    results += [m(labels, estimador[-1].labels_) for m in metricas]

    #Se hace la evaluacion de la distancia inter-cluster e intra-cluster
    results += [
        metrics.silhouette_score(data, estimador[-1].labels_, metric="euclidean", sample_size=200)
        ]
    #Para ponerlo en formato de tabla
    formato_tabla = (
        "{:10s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formato_tabla.format(*results))
print("\n\tTABLA COMPARATIVA DE LAS TÉCNICAS DE CLUSTERING")
print("Técnica\t\tTiempo\tinertia\tHomo\tCompl\tMeasS\tARand\tAdMut\tSilhouette")   
   
### Vamos a utilizar la funcion que evalua con el método Kmeans
#Para esto dividimos los datos con diferentes tecnicas, utlizaremos k-means++, Principal Components Analysis PCA
#y de manera aleatoria
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#La primera evaluacion es de la tecnica PCA con 2 ciclos
pca = PCA(n_components=nWine).fit(data)#ya tenemos un modelo de PCA en scikit-learn
kmeans = KMeans(init=pca.components_, n_clusters=nWine, n_init=1)
Evalua_kmeans(kmeans=kmeans, nombre="PCA-based", data=data, labels=labels)

#La segunda evaluacion es de la tecnica k-means ++ con 2 ciclos
kmeans = KMeans(init="k-means++", n_clusters=nWine, n_init=8,random_state = 0)
Evalua_kmeans(kmeans=kmeans, nombre="k-means++", data=data, labels=labels)

#La tercera evaluacion es de la forma de clustering aleatoria con 2 ciclos
kmeans = KMeans(init="random", n_clusters=nWine, n_init=1,random_state = 0)
Evalua_kmeans(kmeans=kmeans, nombre="random", data=data, labels=labels)

##### VISUALZACION
### Ahora vamos a visualizar los datos en clusters 
import matplotlib.pyplot as plt

#Ajustamos al modelo del PCA porque es la que mejores resultados tuvo
PCAresults = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init="random", n_clusters=nWine, n_init=4)
kmeans.fit(PCAresults)

#el tamaño de h es la celda minima
h = 0.5

#establecemos la vecindad o entorno en la cual se va a decidir si un punto esta dentro o fuera de un cluster
#Esquinas
x_min = PCAresults[:, 0].min() - 1
x_max = PCAresults[:, 0].max() + 1
y_min = PCAresults[:, 1].min() - 1
y_max = PCAresults[:, 1].max() + 1
#Malla
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#REALIZAMOS LA PREDICCION
#con Kmean predecimos el cluster al que sera asignado un punto dentro de la malla "aplanada" (por ravel)
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

#Realizamos las gráficas 
Z = Z.reshape(xx.shape) #Z toma se escala de acuerdo a la malla
plt.figure(1)
plt.clf()
#se grafican los clusters
plt.imshow(Z,interpolation="nearest",extent=(xx.min(), xx.max(), yy.min(), yy.max()),cmap="viridis",aspect="auto",origin="lower")
#se grafican los puntos
plt.plot(PCAresults[:, 0], PCAresults[:, 1], "k*", markersize=3)

#se grafican los centroides
centroides = kmeans.cluster_centers_#este método extrae los centroides
plt.scatter(centroides[:, 0],centroides[:, 1],marker=".",s=180,linewidths=5,color="r",zorder=10)
plt.title( "Clusterización k-means para los datos de pruebas quimica en Vinos\n")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()


