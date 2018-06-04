import numpy as np 
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter

# Es el estilo con el que quiero plotear
style.use('fivethirtyeight')

# este es un diccionario con el dos grupos de datos
# Esta dividido asi para poder usar la primer parte como colores
dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}

# este es el nuevo que quiero saber a donde pertenece
new_features = [5,7]

# itero y ploteo todo el diccionario
for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0],ii[1], s=100, color=i)

# El problema de KNN es que tiene que comprobar cada item como cada uno
# de los items del resto. a veces se pone un radio
# defino la funcion
def k_nearest_neighbors(data, predict, k=3):

    # se fija que haya menos grupos que la variable k,
    # sino no sabria donde ponerlo
    if len(data) >= k:
        # manda un warning
        warnings.warn("K is set to a value")

    # creo una lista con las distancias
    distances = []

    # Itero sobre los grupos
    for group in data:
        # Itero sobre la features
        for features in data[group]:
            # calculo la distancia euclediana con cada feature y el
            # valor a predecir
            eucliden_distance = np.linalg.norm(np.array(features) - np.array(predict))
            # apendeo una lista con la distancia y el grupo
            distances.append([eucliden_distance, group])

    # Armo una lista con los grupos de los tres primeros items de la lista
    votes = [i[1] for i in sorted(distances)[:k]]
    # Counter(votes).most_common(1) devuelve una lista de tuples
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result

result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)
    