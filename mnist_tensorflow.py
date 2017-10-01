import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

n_hidden_1=800
n_hidden_2=800

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
W1=tf.Variable(tf.truncated_normal([784,n_hidden_1]))
b1=tf.Variable(tf.zeros(n_hidden_1))
W2=tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2]))
b2=tf.Variable(tf.zeros(n_hidden_2))
W3=tf.Variable(tf.truncated_normal([n_hidden_2,10]))
b3=tf.Variable(tf.zeros(10))

layer1=tf.add(tf.matmul(x,W1),b1)
layer1=tf.nn.relu(layer1)
layer2=tf.add(tf.matmul(layer1,W2),b2)
layer2=tf.nn.relu(layer2)
out=tf.add(tf.matmul(layer2,W3),b3)
out=tf.nn.sigmoid(out)

cost=tf.reduce	

print("Successful")