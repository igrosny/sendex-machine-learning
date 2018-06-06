import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

# Aca defino la clase
class Support_Vector_Machine:
    # Este es el constructor
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # Esta es la funcion para entrenar
    def fit(self, data):
        pass

    # Esta es la funcion para
    def predict(self, features):
        # sign (x.w+b)
        classification = np.sign(np.dot(np.array(features), self.w)+self.b)
        return classification



# Creamos un diccionario
data_dict = {-1:np.array([[1,6],
                        [2,8],
                        [3,8],]),
            1:np.array([[5,1],
                        [6, -1],
                        [7,3],])}