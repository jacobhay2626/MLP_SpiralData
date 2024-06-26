import numpy as np
import nnfs
from Spiral_Data import create_data

"""
softmax_outputs = [[0.7, 0.1, 0.2],
                   [0.1, 0.5, 0.4],
                   [0.02, 0.9, 0.08]]

class_targets = [0, 1, 1]

for targ_index, distribution in zip(class_targets, softmax_outputs):
    print(distribution[targ_index])
# does not return a vector/array
# confidence for target class index 0 in first row = 0.7
# confidence for target class index 1 in second = 0.5
# confidence for target class index 1 in third = 0.9

"""
# Easier with numpy
"""
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = [0, 1, 1]
"""
# print(softmax_outputs[[0, 1, 2], class_targets])

# print(softmax_outputs[
#           range(len(softmax_outputs)), class_targets
#       ])

# print(-np.log(softmax_outputs[
#           range(len(softmax_outputs)), class_targets
#       ]))
"""
neg_log = -np.log(softmax_outputs[
           range(len(softmax_outputs)), class_targets
       ])

average_loss = np.mean(neg_log)
print(average_loss)
"""
# Handling log(0)
# log(0) gives a loss that would be infinite. Can clip the values in the range by a very small
# number. (1e-7), going to clip all predicted values by same amount.

# y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

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


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # scalar class values
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # one-hot encoded
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


X, y = create_data(100, 3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_SoftMax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss: ", loss)