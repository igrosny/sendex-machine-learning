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

# ploteo el nueo feature
plt.scatter(new_features[0],new_features[1], s=100, color='b')

# muestro el plot
plt.show()


    