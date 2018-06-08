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
            # crea un objeto figure
            self.fig = plt.figure()
            # Le agrega los ejes
            self.ax = self.fig.add_subplot(1, 1, 1)

    # Esta es la funcion para entrenar
    def fit(self, data):
        # data es un diccionario con dos clases -1 y 1
        self.data = data
        #{ ||w||: [w,b]}
        # Este diccionario tiene como clave la distancia
        # y una lista con los valores w y b
        opt_dict = {}

        transforms = [[1,1],
                    [-1,1],
                    [-1,-1],
                    [1,-1],]

        # busca maximo y minimos ranges
        all_data = []
        # Itero los grupo -1 y 1
        for yi in self.data:
            # itero las filas
            for featureset in self.data[yi]:
                #itero las columna
                for feature in featureset:
                    # pongo cada valor en la lista
                    all_data.append(feature)

        # agarro el valor maximo de todas la lista
        self.max_feature_value = max(all_data)
        # agarro el valor minimo
        self.min_feature_value = min(all_data)

        # limpio esto
        all_data = None

        # Empieza con big steps
        step_sizes = [self.max_feature_value * 0.1,
                    self.max_feature_value * 0.01,
                    self.max_feature_value * 0.001,]


        # Extremely expensive, b no tiene que hacer pasos tan peque~no
        # no tiene que ser preciso
        b_range_multiple = 5

        #
        b_multiple = 5

        # selecciono el maximo valor y lo multiplico por 10
        latest_optimum = self.max_feature_value * 10

        # Itero cada step (son 3)
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            # cuando converge
            optimized = False
            while not optimized:
                # np.arange crea in intervalo de espacios iguales
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                        self.max_feature_value * b_range_multiple,
                        step * b_multiple):
                    # lo multipla por cada transformacion
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        # Es eslavon mas fragil del SVM
                        # SMO - ver que es
                        # yi(xi.w + b) >= 1
                        # Itero la clase
                        for i in self.data:
                            #itero las filas
                            for xi in self.data[i]:
                                yi = i # si la clase es negative multiplica por -1
                                # Si i es positivo mutlplica por 1
                                if not yi * (np.dot(w_t , xi) + b) >= 1:
                                    found_option = False
                                    break

                        if found_option:
                            # Calculo la norm del vector
                            # que la norm es la longitud dle vector
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimized = True 
                    print('Optimized a step')
                else:
                    # Le resto un paso
                    w = w - step

            # todas las longitudes ordenadas de menors a mayo
            norms = sorted([n for n in opt_dict])
            # selecciono los valores del optdi
            opt_choice = opt_dict[norms[0]]

            self.w = opt_choice[0]
            self.b = opt_choice[1]

            print(opt_choice)
            latest_optimum = opt_choice[0][0] + step * 2

        

    # Esta es la funcion para predecir
    def predict(self, features):
        # sign (x.w+b)
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)

        # Dibuja el punto a predecir
        if classification !=0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*',c=self.colors[classification])
        
        return classification

    def visualize(self):
        # Pone un punto por cada valor del dict
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # Crea la funcion para calcular el hiperplano
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # positive suppor vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b,1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b,1)
        self.ax.plot([hyp_x_min, hyp_x_max],[psv1,psv2])
        
        # negativ suppor vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b,-1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b,-1)
        self.ax.plot([hyp_x_min, hyp_x_max],[nsv1,nsv2])
        
        # positive suppor vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b,0)
        db2 = hyperplane(hyp_x_max, self.w, self.b,0)
        self.ax.plot([hyp_x_min, hyp_x_max],[db1,db2])

        plt.show()
        


# Creamos un diccionario
data_dict = {-1:np.array([[1,6],
                        [2,8],
                        [3,8],]),
            1:np.array([[5,1],
                        [6, -1],
                        [7,3],])}

svn = Support_Vector_Machine()
svn.fit(data_dict)

predict_us = [[0,10],
            [1,3],
            [3,4],
            [3,5],
            [5,5],
            [5,6],
            [6,-5],
            [5,8]]

for p in predict_us:
    svn. predict(p)     

svn.visualize()
