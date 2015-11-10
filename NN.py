
import numpy
import theano
import theano.tensor as T

class HiddenLayer(object):

	def __init__(self, input_size, output_size, Theta=None):
		self.input_size = input_size
		self.output_size = output_size
		if Theta is None:
			self.Theta = self.initializeWeights(output_size, input_size+1)
		else:
			self.Theta = Theta
		self.z_for_back_prop = None
		self.a_for_back_prop = None
		
		self.Theta_grad = numpy.zeros(self.Theta.shape)


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

	def calculateCost(self, X_input, y_vec, lambda_val):
		m = X_input.shape[0]
		
		h = self.propagateForwardForCost(X_input, y_vec, m)
		
		#Calculate cost
		#(1/m) * sum(sum((-yVec' .* log(h)) - ((1-yVec)' .* log(1 - h))));
		J = T.sum(T.sum(((-y_vec).T * T.log(h)) - ((1-y_vec).T * T.log(1-h))))
		J = J.eval()/m
		#print('J is  ', J)
		#Regularization
		reg_term_sum = 0
		for hidden_layer in self.hidden_layers:
			Theta_with_out_bias = numpy.delete(hidden_layer.Theta, 0, 1)
			reg_term_sum = reg_term_sum + T.sum(T.sum(numpy.square(Theta_with_out_bias))).eval()

		J_Reg_Val = (lambda_val * reg_term_sum)/(2*m);

		J = J+J_Reg_Val;
		print('cost function - forward propagation completed...')
		
		return J

	#this function will add add bias using as a first column
	def addBiasToX(self, X, m):
		result = numpy.append(numpy.ones((m, 1)), X, axis=1)
		return result

	def removeBias(self, X):
		X = X[:,1:]
		return X

	def propagateForwardForCost(self, X_input, y_vec, m):
		layer_idx=0
		prev_layer_output=None
		for hidden_layer in self.hidden_layers:
			if layer_idx==0:
				z_k = T.dot(X_input, hidden_layer.Theta.T)
				prev_layer_output = theano.tensor.nnet.sigmoid(z_k)
			else:
				prev_layer_output = prev_layer_output.eval()
				prev_layer_output = self.addBiasToX(prev_layer_output, m)
				prev_layer_output = T.dot(prev_layer_output, hidden_layer.Theta.T)
				prev_layer_output = theano.tensor.nnet.sigmoid(prev_layer_output)
				
			layer_idx = layer_idx+1
		h = prev_layer_output.eval();

		return h
		
	def propagateForward(self, X_input, y_vec, m):
		
		layer_idx=0
		prev_layer_output=None
		a_k = None
		for hidden_layer in self.hidden_layers:
			if layer_idx==0:
				z_k = numpy.dot(hidden_layer.Theta, X_input)
				prev_layer_output = theano.tensor.nnet.sigmoid(z_k)
			else:
				prev_layer_output = prev_layer_output.eval()
				#Add bias unit
				prev_layer_output = numpy.append(numpy.ones((m, 1)), prev_layer_output, axis=0)
				#save a_k for back propagation
				self.hidden_layers[layer_idx-1].a_for_back_prop = prev_layer_output
				
				#prev_layer_output = self.addBiasToX(prev_layer_output, m)
				z_k = numpy.dot(hidden_layer.Theta, prev_layer_output)
				
				prev_layer_output = theano.tensor.nnet.sigmoid(z_k)

			
			#you need not save when calculating cost as we not dot do 
			#back-propagation during cost calculation.
			hidden_layer.z_for_back_prop = z_k
			
			layer_idx = layer_idx+1

		h = prev_layer_output.eval();
		self.hidden_layers[layer_idx-1].a_for_back_prop = h

		return h

	def propagateBack(self, a_1, delta_final_layer):
		idx = 0
		delta_prev = delta_final_layer
		dt = None
		#skip last layer as as delta(error) at final layer is already calcualted
		for hidden_layer_idx in range(len(self.hidden_layers)-1, -1, -1):
			hidden_layer = self.hidden_layers[hidden_layer_idx]

			if idx == 0:
				#print(delta_prev.shape,  self.hidden_layers[hidden_layer_idx-1].a_for_back_prop.shape)
				dt = numpy.dot(delta_prev, self.hidden_layers[hidden_layer_idx-1].a_for_back_prop.T)
			else:
				z_with_bias = numpy.append(numpy.ones((1, 1)), hidden_layer.z_for_back_prop, axis=0)
				
				delta_prev = T.dot(self.hidden_layers[hidden_layer_idx+1].Theta.T, delta_prev) * self.sigmoidGradient(z_with_bias)
		
				delta_prev = delta_prev.eval()
				#remove bias
				delta_prev = delta_prev[1:,:]
				
				#We need to calculate error at first layer using input layer(i.e a_1)
				if hidden_layer_idx == 0:
					dt = numpy.dot(delta_prev, a_1.T)
				else:
					#This block need to be tested for more than one hidden layer.
					dt = numpy.dot(delta_prev, hidden_layer.a_for_back_prop.T)
					#dt = numpy.dot(delta_prev, self.hidden_layers[hidden_layer_idx-1].a_for_back_prop.T)
			
			hidden_layer.Theta_grad = hidden_layer.Theta_grad + dt
			idx = idx+1
		return None
	
	def mean_and_regularize_theta(self, m, lambda_val):
		for hidden_layer in self.hidden_layers:
			hidden_layer.Theta_grad = hidden_layer.Theta_grad/m
			print(hidden_layer.Theta_grad.shape)
			hidden_layer.Theta_grad[:,1:] = hidden_layer.Theta_grad[:,1:] + lambda_val * (hidden_layer.Theta[:,1:]/m)
			print(hidden_layer.Theta_grad.shape)
			
		
		
	def sigmoidGradient(self, val):
		sig_val = theano.tensor.nnet.sigmoid(val)
		sgd = sig_val * (1-sig_val)
		return sgd.eval()
	
	def ignite(self, X, y_vec, lambda_val):
		m = X.shape[0]
		X = self.addBiasToX(X, m)
		
		cost = self.calculateCost(X, y_vec, lambda_val)
		print('calculated cost is: ', cost)
		
		X_col_length = X.shape[1]
		y_row_length = y_vec.shape[0]
		for data_idx in range(0, m):
			X_one_rec = numpy.reshape(X[data_idx,:], (X_col_length, 1))
			y_one_rec = numpy.reshape(y_vec[:,data_idx], (y_row_length,1))
			a_final = self.propagateForward(X_one_rec, y_one_rec, 1)
			
			#Calculate error at final layer
			delta_final_layer = a_final - y_one_rec
			
			self.propagateBack(X_one_rec, delta_final_layer)
			
			self.mean_and_regularize_theta(m, lambda_val)
			
