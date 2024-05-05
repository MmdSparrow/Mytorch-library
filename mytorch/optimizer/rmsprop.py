from mytorch import Tensor
from mytorch.optimizer import Optimizer


import numpy as np

"TODO: (optional) implement RMSprop optimizer"
class RMSprop(Optimizer):
    def __init__(self, layers, learning_rate=0.001, rho=0.9, epsilon=1e-8):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.accumulators = {}
        for layer_index, layer in enumerate(self.layers):
            self.accumulators[(layer_index, "w")] = Tensor(np.zeros_like(layer.weight.data))
            if layer.need_bias:
                self.accumulators[(layer_index, "b")] = Tensor(np.zeros_like(layer.bias.data))

    def step(self):
        layer_index = -1
        for layer in self.layers:
            layer_index+=1
            if layer.weight is not None and layer.weight.requires_grad:
                self.accumulators[(layer_index, "w")] = self.rho * self.accumulators[(layer_index, "w")] + (1 - self.rho) * (layer.weight.grad ** 2)
                update = self.learning_rate * layer.weight.grad * (((self.accumulators[(layer_index, "w")]).__pow__(1/2)) + self.epsilon).__pow__(-1)
                layer.weight = layer.weight - update

            if layer.need_bias and layer.bias is not None and layer.bias.requires_grad:
                self.accumulators[(layer_index, "b")] = self.rho * self.accumulators[(layer_index, "b")] + (1 - self.rho) * (layer.bias.grad ** 2)
                update = self.learning_rate * layer.bias.grad * (((self.accumulators[(layer_index, "b")]).__pow__(1/2)) + self.epsilon).__pow__(-1)
                layer.bias = layer.bias - update
