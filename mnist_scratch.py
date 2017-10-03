import numpy as np
import cv
import cv2

data=np.genfromtxt("MNIST_data/test.csv",delimiter=',')
rows=np.array([i for i in range(1,data.shape[0])])
cols=np.array([i for i in range(1,data.shape[1])])
X_=data[rows,:]
X=X_[:,cols]
Y=data[rows,1]

def relu(layer):
	for i in range(0,layer.shape[0]):
		for j in range(0,layer.shape[1]):
			if(layer[i][j]<0):
				layer[i][j]=0;
	return layer

def sigmoid(layer):
	return (1.0/(1.0-np.exp(-1*layer)))

hidden_1=800
hidden_2=800
num_iter=1

W1=np.random.rand(784,hidden_1)*0.1
b1=np.zeros((1,hidden_1))
W2=np.random.rand(hidden_1,hidden_2)*0.1
b2=np.zeros((1,hidden_2))
W3=np.random.rand(hidden_2,10)*0.1
b3=np.zeros((1,10))

for i in range(0,num_iter):
	layer_1=np.add(np.dot(X,W1),b1)
	layer_1=relu(layer_1)
	layer_2=np.add(np.dot(layer_1,W2),b2)
	layer_2=relu(layer_2)
	out=np.add(np.dot(layer_2,W3),b3)
	out=sigmoid(out)
	