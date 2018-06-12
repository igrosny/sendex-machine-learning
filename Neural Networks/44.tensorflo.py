import tensorflow as tf

x1 = tf.constant(5.0)
x2 = tf.constant(6.0)

#no pasa nada hasta que correr la info
result = tf.multiply(x1, x2)

print(result)

# sess = tf.Session()
# print(sess.run(result))
# # siempre hay que cerrar
# sess.close() 

with tf.Session() as sess:
    output = sess.run(result)
    print(output)

print(output)