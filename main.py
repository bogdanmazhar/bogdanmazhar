import numpy as np

inputs = [5, 20, 70, 3, 8]
weights = [-0.4, 0.7, 0.5, 0.4, 0.3]
bias = -0.5
weighted_inputs = [inputs[i] * weights[i] for i in range(len(inputs))]
total_input = sum(weighted_inputs) + bias

def relu(x):
    return max(0, x)
activated_output = relu(total_input)
print("Рейтинг игрока:", activated_output)