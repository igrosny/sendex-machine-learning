import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

# Leo el archivo de datos y lo pongo en un dataframe
df = pd.read_csv('breast-cancer-wisconsin.data.txt')

# Segun la documentacion todos los datos faltantes tiene '?'
# los remplazamos con un numero grande
df.replace('?',-99999, inplace=True)

# Elimino la columna id que no sirve para nada
df.drop(['id'], 1, inplace=True)

# Convierto a un arreglo todo menos la columna class
X = np.array(df.drop(['class'], 1))
# Guardo la columna class como el label
y = np.array(df['class'])

# Separo los datos en lo que quiero entrenar y con lo 
# que quiero testear
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# Creo el classifier
clf = neighbors.KNeighborsClassifier()

# Entreno el classifier
clf.fit(X_train, y_train)

# Calculo su resultado
accuracy = clf.score(X_test, y_test)
print(accuracy)

# Esta seccion es para predecir
example_measures = np.array([4,2,1,1,1,2,3,2,1])
# La siguiente lines la tuve que poner por compatibilidad
example_measures = example_measures.reshape(1, -1)

# Y con esto se predice
prediction = clf.predict(example_measures)
print(prediction)