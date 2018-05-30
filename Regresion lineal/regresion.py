import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style

# esto es simplemente para que los graficos se vean bonitos
style.use('ggplot')

# La librearia quand trae el historial diario en este caso de las
# acciones de google
# Aca tenes como se instala https://www.quandl.com/tools/python
df = quandl.get("WIKI/GOOGL")
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

# HL_PCT significa High - Low percentage y es el procentaje de variacion
# entre el pico del dia y la parte mas baja
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0

# PCT_change es el porcentaje de cambio
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

# Aca filtramos solo las columnas que nos interesan
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# declaro la columna que quiero predecir
forecast_col = 'Adj. Close'

# Completo los valores nulos de la tabla con -99999 que se van a tomar
# como outliers, inplace significa que cambia los datos en df
df.fillna(value=-99999, inplace=True)

# Detallo cuanto voy a predecir. En este caso es el 1% del tamaño del dataframe
forecast_out = int(math.ceil(0.01 * len(df)))

# Lable es la etiqueta que quiero pronosticar, en este caso es el valor de cierre
# y la copio pero corro toda la columna forcast_out para "abajo"
# Basicamente los ultimos valores de la columna son Nan
df['label'] = df[forecast_col].shift(-forecast_out)

# Es convencion llamar a las features 'X' y a los labels 'y'
# np.array convierte el dataframe a un array 
# el uno es para columna
X = np.array(df.drop(['label'], 1))

# convierte todos lo valores en un rango de -1 a 1
X = preprocessing.scale(X)

# Copio las ultimas filas que no tiene valores de label
X_lately = X[-forecast_out:]
# Me quedo con todos los valores menos lo que saqure antes
X = X[:-forecast_out]

# Eliminas las filas de cuando falta al menos un elemento
df.dropna(inplace=True)

# guardo los labels
y = np.array(df['label'])


# Separo los valores que quiero entrenar
# y con los que voy a testear mas adelante
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# Defino el classifier
clf = LinearRegression()

# Entreno el classifier
clf.fit(X_train, y_train)

# lo testeamos usando la data para testing
confidence = clf.score(X_test, y_test)

print(confidence)

# Se le pasa el array a predecir
forecast_set = clf.predict(X_lately)

# Creo una columna vacias para predecir
df['Forecast'] = np.nan

# iloc es index location, en este caso te devuelve el id
# de la ultima fila
last_date = df.iloc[-1].name
# consigo la ultima fecha
last_unix = last_date.timestamp()
# cantidad de segundos en un dia
one_day = 86400
# Le suma el siguiente dia
next_unix = last_unix + one_day

# itera en este caso por los 35 dias
for i in forecast_set:
    # La libreria tiena una clase que tiene un metodo que convierte unix en fecha
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

# Grafico el valor de cierre
df['Adj. Close'].plot()
# Grafico la columna de prediccion
df['Forecast'].plot()

plt.legend(loc=4)
# etiquetas en el grafico
plt.xlabel('Date')
plt.ylabel('Price')

#muestro el grafico
plt.show()