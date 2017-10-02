import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

n_hidden_1=800
n_hidden_2=800

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
W1=tf.Variable(tf.random_normal([784,n_hidden_1])*0.01)
b1=tf.Variable(tf.zeros(n_hidden_1))
W2=tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])*0.01)
b2=tf.Variable(tf.zeros(n_hidden_2))
W3=tf.Variable(tf.random_normal([n_hidden_2,10])*0.01)
b3=tf.Variable(tf.zeros(10))

layer1=tf.add(tf.matmul(x,W1),b1)
layer1=tf.nn.relu(layer1)
layer2=tf.add(tf.matmul(layer1,W2),b2)
layer2=tf.nn.relu(layer2)
out=tf.add(tf.matmul(layer2,W3),b3)
out=tf.nn.sigmoid(out)

cost=tf.reduce_mean(tf.reduce_sum(-y*tf.log(out)-(1-y)*tf.log(1-out),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

sess=tf.InteractiveSession()
tf.global_variables_initializer().run()

# for i in range(10000):
# 	batch_xs,batch_ys=mnist.train.next_batch(100)
# 	sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
# 	if(i%50==0):
# 		print(i)

#print("Successfully trained\n")

# saver=tf.train.Saver([W1,b1,W2,b2,W3,b3])
# save_path=saver.save(sess,"/home/sanskar/DL/mnist/restore.ckpt")
# print("saved in ",save_path)
# print("Successfully saved\n")

saver=tf.train.Saver()#[W1,b1,W2,b2,W3,b3])
saver.restore(sess,"restore.ckpt")
print("Restored Successfully")


predicted=tf.argmax(out,1)+1
correct=tf.argmax(y,1)+1
error=(predicted-correct)*100
print(sess.run(cost,feed_dict={x:mnist.test.images,y:mnist.test.labels}))
print(predicted.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels}))
print(tf.argmax(y,1).eval(feed_dict={x:mnist.test.images,y:mnist.test.labels})+1)

correct_prediction=tf.equal(tf.argmax(out,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))*100	
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))

sess.close()
