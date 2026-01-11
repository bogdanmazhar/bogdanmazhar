import numpy as np
inputs = [37, 59, 25]
weights = [0.4, 0.4, 0.2]
bias = -10
weighted_inputs = [inputs[i] * weights[i] for i in range(len(inputs))]
total_input = sum(weighted_inputs) + bias
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
activated_output = sigmoid(total_input)
print("Настроение космиеского исследователя:", activated_output)