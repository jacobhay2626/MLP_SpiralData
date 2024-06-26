# want to measure how wrong an output is.
# In the output layer, what if both our outputs are negative? Both of the outputs are going to be set to zero.
# exponential function: makes sure no value can be negative.
# normalisation: gives probability distribution we want.
# this exponential then normalisation is the softmax activation function.

# An issue with using exponents is the explosion of values as the input grow, leading to overflow. A way around this
# is to minus every element in the output layer prior to exponention by the largest value in the vector.
# Now the largest value is going to be zero, and this means our range of possibilities is between 0 and 1.
# The output is exactly the same after normalisation, just that we have prevented any overflow!
"""
import numpy as np
import nnfs

nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)
print(exp_values)

# axis = none (sum of all values)
# axis = 0 (sum of all columns)
# axis = 1 (sum of rows)
# keepdims to make a column vector

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
print(norm_values)
"""
import numpy as np
import nnfs
from Spiral_Data import create_data

nnfs.init()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights + self.biases)


class Activation_ReLU:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_SoftMax:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


X, y = create_data(100, 3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3)
activation2 = Activation_SoftMax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])