import numpy as np 
from math import sqrt
import warnings
from collections import Counter
import random
import pandas as pd 

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

# Levanto el archivo con datos
df = pd.read_csv("breast-cancer-wisconsin.data.txt")

# Replazo los valores faltantes
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

# astype(float) lo convierte todos los valores en float
# values lo convierte en un numpy array
# tolist: return a copy of the array data as nested python list
full_data = df.astype(float).values.tolist()

# randomizes the items of a list in place
random.shuffle(full_data)

# defino el tamano del test
test_size = 0.2
# defino los grupos para entrenar
train_set = {2:[], 4:[]}
# Defino el grupo para testear
test_set = {2:[], 4:[]}
# Separo la data, desde el principio hasta (el final menos el x%)
train_data = full_data[:-int(test_size * len(full_data))]
# separo los datos con los que quiero testear
test_data = full_data[-int(test_size * len(full_data)):]

# Itero la data de entrenamiento
for i in train_data:
    # de fila i, la columna -1 es la ultima que en 
    # este caso es la clase, a ese subgrupo le agrego toda
    # la fila i menos la ultima columna
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

# defino variables en 0
correct = 0
total = 0

# Itero los grupos en test set
for group in test_set:
    # Itero las filas en cada grupo
    for data in test_set[group]:
        # Calculo el grupo con knn  y los datos de entrenamiento
        vote = k_nearest_neighbors(train_set, data, k=5)
        # comparo y si es igual le sumo uno
        if group == vote:
            correct += 1
        total += 1

print('Accuracy: ', correct/total)
