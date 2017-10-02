import tensorflow as tf

#with tf.Session(config = config) as sess:

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

n_hidden_1=1000
n_hidden_2=1000
n_hidden_3=1000

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
W1=tf.Variable(tf.random_normal([784,n_hidden_1],stddev=0.1))
b1=tf.Variable(tf.constant(0.1,shape=[n_hidden_1]))
W2=tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2],stddev=0.1))
b2=tf.Variable(tf.constant(0.1,shape=[n_hidden_2]))
W3=tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3],stddev=0.1))
b3=tf.Variable(tf.constant(0.1,shape=[n_hidden_3]))
W4=tf.Variable(tf.random_normal([n_hidden_2,10],stddev=0.1))
b4=tf.Variable(tf.constant(0.1,shape=[10]))

layer1=tf.add(tf.matmul(x,W1),b1)
layer1=tf.nn.relu(layer1)
layer2=tf.add(tf.matmul(layer1,W2),b2)
layer2=tf.nn.relu(layer2)
layer3=tf.add(tf.matmul(layer2,W3),b3)
layer3=tf.nn.relu(layer3)
out=tf.add(tf.matmul(layer3,W4),b4)
#out=tf.nn.sigmoid(out)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=out))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

correct_prediction=tf.equal(tf.argmax(out,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))*100	

sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.reset_default_graph()



#File_Writer=tf.summary.FileWriter('./Graph',sess.graph)

# #*********For Training and saving***********
# for i in range(10000):
# 	batch_xs,batch_ys=mnist.train.next_batch(50)
# 	sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
# 	if(i%100==0):
# 		print(i)
# 		#print(accuracy.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels}))
# print("Successfully trained\n")
# saver=tf.train.Saver([W1,b1,W2,b2,W3,b3,W4,b4])
# save_path=saver.save(sess,"/home/sanskar/DL/mnist/restore.ckpt")
# print("saved in ",save_path)
# print("Successfully saved\n")

#*******Restoring************
saver=tf.train.Saver([W1,b1,W2,b2,W3,b3,W4,b4])
saver.restore(sess,"restore.ckpt")
print("Restored Successfully")


predicted=tf.argmax(out,1)+1
print(sess.run(cost,feed_dict={x:mnist.test.images,y:mnist.test.labels}))
print('predicted = ',predicted.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels}))
print('original =  ',tf.argmax(y,1).eval(feed_dict={x:mnist.test.images,y:mnist.test.labels})+1)
print("Accuracy : ",sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))

sess.close()
