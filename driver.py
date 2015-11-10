import numpy
import theano

from DataLoader import load_data
from NN import NNetwork

def create_y_vec(y, classes):
	m = len(y)
	for i in range(0, m):
		if y[i]==0:
			y[i] = 10
		#print(y[i])
	
	y_vec = numpy.zeros(
			(classes, m),
			dtype=theano.config.floatX
			)
	
	for i in range(0, m):
		y_vec[y[i]-1][i] = 1

	return y_vec

def drive(input_layer_size, hidden_layer_size, classes, is_local):
	
	X, y = load_data('../data/train_sample.csv', is_local, need_split=False)
	
	y_vec = create_y_vec(y, classes)
	lambda_val = 0.1
	#print(y_vec)
	network = NNetwork(input_layer_size, hidden_layer_size, 1, classes)
	network.ignite(X, y_vec, lambda_val)

if __name__ == '__main__':
	input_layer_size = 784
	hidden_layer_size = 392
	classes = 10
	is_local = True
	
	drive(input_layer_size, hidden_layer_size, classes, is_local)
