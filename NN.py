
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


	def initializeWeights(self, input_size,output_size):
		epsilon_init = numpy.sqrt(6. / (input_size + output_size))
		rng = numpy.random.RandomState(1234)

		weights = numpy.asarray(
				rng.uniform(
					low=-epsilon_init,
					high=epsilon_init,
					size=(input_size+1, output_size)
				),
				dtype=theano.config.floatX
			)
		#print(input_size, output_size)
		return weights

	def process(self, data_x):
		#print('HiddenLayer.....%i...%i', self.Theta.shape, data_x.shape)
		#data_x = T.transpose(data_x)
		#print(data_x.T)
		result = numpy.dot(self.Theta, data_x.T)
		return result
		#z_2 = self.Theta * self.data_x;

class NNetwork(object):
	def __init__(self, input_layer_size, hidden_layer_size, no_of_hidden_layes, classes):
		self.input_layer_size = input_layer_size
		self.hidden_layer_size = hidden_layer_size
		self.no_of_hidden_layes = no_of_hidden_layes
		self.classes = classes
		self.hidden_layers = self.init_hidden_layers(input_layer_size, hidden_layer_size, no_of_hidden_layes, classes)


	#initialize all hidden layers with their respective input and output sizes.		
	def init_hidden_layers(self, input_layer_size, hidden_layer_size, no_of_hidden_layes, classes):
		layers = []
		for layer in range(1, no_of_hidden_layes + 2):
			#input/first layer
			if layer==1:
				layers.append(HiddenLayer(input_layer_size, hidden_layer_size, None))
			#output/last layer
			elif layer == (no_of_hidden_layes + 1):
				layers.append(HiddenLayer(hidden_layer_size, classes, None))
			else:
				layers.append(HiddenLayer(hidden_layer_size, hidden_layer_size, None))
		return layers

	def calculateCost(self, X_input, y_vec):
		m = X_input.shape[0]
		x_mat  = T.matrix('x_mat')
		
		layer_idx=0
		prev_layer_output=None
		for hidden_layer in self.hidden_layers:
			if layer_idx==0:
				z_k = T.dot(X_input, hidden_layer.Theta)
				prev_layer_output = theano.tensor.nnet.sigmoid(z_k)
			else:
				prev_layer_output = self.addBiasToX(prev_layer_output.eval(), m)
				prev_layer_output = T.dot(prev_layer_output, hidden_layer.Theta)
				prev_layer_output = theano.tensor.nnet.sigmoid(prev_layer_output)
				
			layer_idx = layer_idx+1
		h = prev_layer_output.eval();
		print(h.shape)
		
		#Calculate cost
		J = T.sum(T.sum(T.dot((-y_vec), T.log(h))) - T.dot((1-y_vec), T.log(1-h)))
		J = J.eval()/m
		
		#Regularization
		reg_term_sum = 0
		for hidden_layer in self.hidden_layers:
			Theta_with_out_bias = numpy.delete(hidden_layer.Theta, 0, 1)
			print(hidden_layer.Theta.shape)
			print(Theta_with_out_bias.shape)
			reg_term_sum = reg_term_sum + T.sum(T.sum(numpy.square(Theta_with_out_bias))).eval()

		lambda_val = 0.1
		J_Reg_Val = (lambda_val * reg_term_sum)/(2*m);

		J = J+J_Reg_Val;
		print('cost function - forward propagation completed...')
		
		return J

	#this function will add add bias using as a first column
	def addBiasToX(self, X, m):
		#print(X.shape, m)
		return numpy.append(numpy.ones((m, 1)), X, axis=1)

	def propagateForward(self, X, y):
		layer_idx = 0;
		prev_layer_output = None
		for hidden_layer in self.hidden_layers:
			if layer_idx==0:
				prev_layer_output = hidden_layer.process(X.loc[train_idx])
			else:
				prev_layer_output = hidden_layer.process(prev_layer_output)
			layer_idx = layer_idx+1

		return prev_layer_output


	def ignite(self, X, y_vec):
		m = X.shape[0]
		X = self.addBiasToX(X, m)
		
		
		cost = self.calculateCost(X, y_vec)
		print('calculated cost is: ', cost)
		#for train_idx in range(0, m):
		#	result = self.propagateForward(X, y_vec)
		#	self.propagateBack()

