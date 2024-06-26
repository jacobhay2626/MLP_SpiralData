# Categorical cross entropy
# One hot encoding.
# Successful, and convenient method in backprop and optimisation steps.
# One hot encoding:
# vector with n classes, filled with zeros except index of target index (1)
# -log(y(i))

import math

softmax_output = [0.7, 0.1, 0.2]

target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])

print(loss)