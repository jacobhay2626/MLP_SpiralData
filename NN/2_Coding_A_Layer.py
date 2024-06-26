# Coding a layer with four inputs and three outputs.

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.9, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]


layer_outputs = []

for neuron_weight, neuron_biase in zip(weights, biases):
    neuron_output = 0
    for n_inputs, weight in zip(inputs, neuron_weight):
        neuron_output += n_inputs*weight
    neuron_output += neuron_biase
    layer_outputs.append(neuron_output)

print(layer_outputs)