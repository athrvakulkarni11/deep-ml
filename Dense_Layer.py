import numpy as np
import math

np.random.seed(42)

class Layer(object):
    def set_input_shape(self, shape):
        self.input_shape = shape

    def layer_name(self):
        return self.__class__.__name__

    def parameters(self):
        return 0

    def forward_pass(self, X, training):
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        raise NotImplementedError()

    def output_shape(self):
        raise NotImplementedError()


class Dense(Layer):
    def __init__(self, n_units, input_shape=None, learning_rate=0.01):
        """
        Initializes a Dense layer.
        Args:
            n_units: Number of neurons in the layer.
            input_shape: Shape of the input data (tuple).
            learning_rate: Learning rate for weight updates.
        """
        self.n_units = n_units
        self.input_shape = input_shape
        self.trainable = True
        self.W = None
        self.w0 = None
        self.learning_rate = learning_rate
        self.layer_input = None
        
        if input_shape:
            self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes the weights and biases of the layer.
        """
        input_dim = self.input_shape[0]
        self.W = np.random.randn(input_dim, self.n_units) * 0.01  # Small random values
        self.w0 = np.zeros((1, self.n_units))  
    def forward_pass(self, X, training=True):
        """
        Performs the forward pass for the Dense layer.
        Args:
            X: Input data (numpy array).
            training: Flag to indicate whether in training mode.
        Returns:
            Output after applying the Dense layer.
        """
        if self.W is None or self.w0 is None:
            self.input_shape = X.shape[1:]
            self._initialize_weights()

        self.layer_input = X
        return np.dot(X, self.W) + self.w0

    def backward_pass(self, accum_grad):
        """
        Performs the backward pass for the Dense layer.
        Args:
            accum_grad: Gradient of the loss with respect to the output of this layer.
        Returns:
            Gradient of the loss with respect to the input of this layer.
        """
        grad_W = np.dot(self.layer_input.T, accum_grad)
        grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)
        grad_input = np.dot(accum_grad, self.W.T)

        if self.trainable:
            self.W -= self.learning_rate * grad_W
            self.w0 -= self.learning_rate * grad_w0

        return grad_input

    def number_of_parameters(self):
        """
        Returns the total number of trainable parameters in the layer.
        """
        return np.prod(self.W.shape) + np.prod(self.w0.shape)


dense_layer = Dense(n_units=4, input_shape=(3,), learning_rate=0.01)

X = np.random.randn(5, 3) 

output = dense_layer.forward_pass(X, training=True)
print("Forward pass output:\n", output)
accum_grad = np.random.randn(5, 4) 

grad_input = dense_layer.backward_pass(accum_grad)
print("Backward pass gradient wrt input:\n", grad_input)

num_params = dense_layer.number_of_parameters()
print("Number of parameters:", num_params)
