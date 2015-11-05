
import numpy
import theano
import theano.tensor as T

class HiddenLayer(object):

	def __init__(self, input_size, output_size, Theta=None):
		self.input_size = input_size
		self.output_size = output_size
		if Theta is None:
			self.Theta = self.initializeWeights(input_size, output_size)
		else:
			self.Theta = Theta
		print(self.Theta.shape)


	def initializeWeights(self, input_size,output_size):
		epsilon_init = numpy.sqrt(6. / (input_size + output_size))
		rng = numpy.random.RandomState(1234)

		weights = numpy.asarray(
				rng.uniform(
					low=-epsilon_init,
					high=epsilon_init,
					size=(input_size, output_size)
				),
				dtype=theano.config.floatX
			)

		return weights
	

class NNetwork(object):
	def __init__(self, input_layer_size, hidden_layer_size, no_of_hidden_layes, classes):
		self.input_layer_size = input_layer_size
		self.hidden_layer_size = hidden_layer_size
		self.no_of_hidden_layes = no_of_hidden_layes
		self.classes = classes
		self.hidden_layes = self.init_hidden_layers(input_layer_size, hidden_layer_size, no_of_hidden_layes, classes)


	#initialize all hidden layers with their respective input and output sizes.		
	def init_hidden_layers(self, input_layer_size, hidden_layer_size, no_of_hidden_layes, classes):
		layers = []
		for layer in range(0, no_of_hidden_layes + 2):
			#input/first layer
			if layer==0:
				layers.append(HiddenLayer(input_layer_size, hidden_layer_size, None))
			#output/last layer
			elif layer == (no_of_hidden_layes + 1):
				layers.append(HiddenLayer(hidden_layer_size, classes, None))
			else:
				layers.append(HiddenLayer(hidden_layer_size, hidden_layer_size, None))
		
		return layers


	def ignite(X, y):
		print('test')
