import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


X, y = spiral_data(100, 3)


'''
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

for i in inputs:
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)
print(output)
or
for i in inputs:
    output.append(max(0, i))
'''


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = None

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = LayerDense(2, 5)
#layer2 = LayerDense(5, 2)
activation1 = ActivationReLU
layer1.forward(X)
# print(layer1.output)
#layer2.forward(layer1.output)
#print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)
