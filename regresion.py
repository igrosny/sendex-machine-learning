import Quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

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
# como outliers
df.fillna(value=-99999, inplace=True)

# Detallo cuanto voy a predecir. En este caso es el 1% del tamaño del dataframe
forecast_out = int(math.ceil(0.01 * len(df)))

# Lable es la etiqueta que quiero pronosticar, en este caso es el valor de cierre
# y la copio pero corro toda la columna forcast_out para "abajo"
# Basicamente los ultimos valores de la columna son Nan
df['label'] = df[forecast_col].shift(-forecast_out)
