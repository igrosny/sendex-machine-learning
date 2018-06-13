import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

"""
input > weight > hidden layer (activation function) > weights > hidden layer 2
(activation function) > weights > output layer 

FEED forward

compare output to intended outpo  > cost function (cross entropy)
optimiation function (optimizer) > minimize cost (AdamOptimizer.. SGD, Adagrad)

back propagation

feed forward + backprop = epoch
"""

# Importo los datos de minst
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# Creo los 3 hidden layers
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# Creo el numero de clases
n_classes = 10

# de a cuantos se va a ir probando
batch_size = 100

# Esta es la parte en donde comienzo a armar el modelo
# defino el tipo de variable de los features
x = tf.placeholder('float', [None, 784])
# defino el tipo de datos de los labels
y = tf.placeholder('float')

# Creo la funcion que crea el modelo
def neural_network_model(data):

    # Comienza con valores random de un distrbucion normal en los weight
    # y lo mismo en lo bises
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}

    # l1 = d * w + b
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    # Computa un rectificador lineal se llaman activation functions
    # sigmoid functions _/ o arctan
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):

    prediction = neural_network_model(x)
    # el output layer usa softmaz si es classification
    # or linear para regression
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    # agrega el otimizer que crea slots para minimizar el cost?
    # para leer mas adelante https://arxiv.org/pdf/1412.6980v8.pdf
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    # How may epochs: cuantos ciclos quiero correr
    hm_epochs = 10
    # Levanto la session con cierre automatico
    with tf.Session() as sess:
        # adds an operation to initialize the variables
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            # el numero de ejemplo sobre el batch size
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        # argmax devuelve el valor maximo
        # equal devuelve un tensor de tipo booleano
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        # devuelve el promedio
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)