from mytorch.optimizer import Optimizer

import numpy as np
from typing import List
import numpy as np

from torch import Tensor
from mytorch.layer import Linear
from mytorch.optimizer import Optimizer


# TODO: implement Adam optimizer like SGD
class Adam(Optimizer):
    def __init__(self, layers: list[Linear], learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, time = 1):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.time = time
        self.momentums = {}
        self.velocities = {}

        for layer_index, layer in enumerate(self.layers):
            self.momentums[(layer_index, "w")] = Tensor(np.zeros_like(layer.weight.data))
            self.velocities[(layer_index, "w")] = Tensor(np.zeros_like(layer.weight.data))
            if layer.need_bias:
                self.momentums[(layer_index, "b")] = Tensor(np.zeros_like(layer.bias.data))
                self.velocities[(layer_index, "b")] = Tensor(np.zeros_like(layer.bias.data))

    def step(self):
        self.time += 1
        beta1_t = self.beta1 ** self.time
        beta2_t = self.beta2 ** self.time

        layer_index= -1
        for layer in self.layers:
            layer_index+=1
            if layer.weight is not None and layer.weight.requires_grad:
                self.momentums[(layer_index, "w")] = self.beta1 * self.momentums[(layer_index, "w")] + (1 - self.beta1) * layer.weight.grad
                self.velocities[(layer_index, "w")] = self.beta2 * self.velocities[(layer_index, "w")] + (1 - self.beta2) * (layer.weight.grad ** 2)
                m_hat = self.momentums[(layer_index, "w")] * (1/(1 - beta1_t))
                v_hat = self.velocities[(layer_index, "w")] * (1/(1 - beta2_t))
                update = self.learning_rate * m_hat * ((v_hat).__pow__(1/2) + self.epsilon).__pow__(-1)
                layer.weight = layer.weight - update

            if layer.need_bias and layer.bias is not None and layer.bias.requires_grad:
                self.momentums[(layer_index, "b")] = self.beta1 * self.momentums[(layer_index, "b")] + (1 - self.beta1) * layer.bias.grad
                self.velocities[(layer_index, "b")] = self.beta2 * self.velocities[(layer_index, "b")] + (1 - self.beta2) * (layer.bias.grad ** 2)
                m_hat = self.momentums[(layer_index, "b")] * (1/(1 - beta1_t))
                v_hat = self.velocities[(layer_index, "b")] * (1/(1 - beta2_t))
                update = self.learning_rate * m_hat * ((v_hat).__pow__(1/2)+ self.epsilon).__pow__(-1)
                layer.bias = layer.bias - update
