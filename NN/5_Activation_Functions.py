# Act Funcs
# Each neuron will have this activation function which comes into play after computing
# the weight*input + bias. That value is fed into the activation function.
# output layer usually has different function to the hidden layer neurones.

# Sigmoid
# Reliable for training the NN. More granular output from this function than the step function.
# Step function doesn't have the granularity to tell us how close our output was to 1 or 0.

# ReLU
# if x > 0 output is x.
# if x <= 0 output is 0.
# Sigmoid AF has the vanishing gradient problem, as ReLU is granular, is fast, and it works. Most popular
# AF for hidden layers.

# Why use AF?
# If we were to just use weights and biases, the activation function would just be a linear plot of y = x.
# With a linear activation function we can only fit linear data, but we mainly use non-linear data.
# If you imagine a sigmoid function, and we try to train a NN to draw this line using a linear AF, it is
# impossible to imagine a straight line fitting that sigmoid. A non-linear activation function can.
# For ReLU, the weight changes the gradient of the line, and the bias moves the function horizontally.

import numpy as np
import nnfs
from Spiral_Data import create_data

nnfs.init()

# X = [[1, 2, 3, 2.5],
#      [2.0, 5.0, -1.0, 2.0],
#      [-1.5, 2.7, 3.3, -0.8]]

X, y = create_data(100, 3)

"""
ReLU
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
outputs = []


for i in inputs:
    outputs.append(max(0, i))

print(outputs)
"""


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


layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer1.forward(X)
# print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)