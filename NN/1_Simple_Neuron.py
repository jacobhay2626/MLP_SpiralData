inputs = [12.5, 6, 8.9]
weights = [3.43, 7.2, 1.23]
bias = 4

output = 0

for n_input, weight in zip(inputs, weights):
    output += n_input*weight
output = output + bias
print(output)
