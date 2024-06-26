# Batches:
# Allow us to use gpu instead of cpu
# Helps with generalisation: our current inputs describe the current status of our server at this point in
# time. We want to pass a batch of these samples instead, showing the algorithm multiple at once, making it more
# likely to generalise. Makes it easier for our neurone to fit the line.
# Too large a batch can cause overfitting, (32, 64)


# Need to initialise weights as random values, want small values to avoid explosion of values.

import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights + self.biases)


layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 6)
layer3 = Layer_Dense(6, 7)
layer4 = Layer_Dense(7, 8)

layer1.forward(X)
# print(layer1.output)
layer2.forward(layer1.output)
# print(layer2.output)
layer3.forward(layer2.output)
# print(layer3.output)
layer4.forward(layer3.output)
print(layer4.output)
"""
# BEFORE USING OBJECTS 

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

# second layer

weights_2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]

biases_2 = [-1, 2, -0.5]

# matrix multiplication
# Need to use the transpose of the weights matrix.

outputs = np.dot(inputs, np.array(weights).T) + biases
print(outputs)


# layer 1 outputs become second layer inputs
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

layer2_outputs = np.dot(layer1_outputs, np.array(weights_2).T) + biases_2

print(layer2_outputs)
"""
