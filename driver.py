
from DataLoader import load_data
from NN import NNetwork

def drive(input_layer_size, hidden_layer_size, classes, is_local):
	
	X, y = load_data('../data/train_sample.csv', is_local, need_split=False)

	network = NNetwork(input_layer_size, hidden_layer_size, 1, classes)
	network.ignite(X, y)
	
	

if __name__ == '__main__':
	input_layer_size = 784
	hidden_layer_size = 392
	classes = 10
	is_local = True
	
	drive(input_layer_size, hidden_layer_size, classes, is_local)
