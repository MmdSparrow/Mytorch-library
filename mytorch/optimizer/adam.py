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
        self.momentums = [np.zeros_like(param) for layer in self.layers for param in layer.parameters()]
        self.velocities = [np.zeros_like(param) for layer in self.layers for param in layer.parameters()]

    def step(self):
        self.time += 1
        beta1_t = self.beta1 ** self.time
        beta2_t = self.beta2 ** self.time

        for layer in self.layers:
            for param_idx, param in enumerate(layer.parameters()):
                param_grad = layer.grads[param_idx]
                self.momentums[param_idx] = self.beta1 * self.momentums[param_idx] + (1 - self.beta1) * param_grad
                self.velocities[param_idx] = self.beta2 * self.velocities[param_idx] + (1 - self.beta2) * (param_grad ** 2)
                m_hat = self.momentums[param_idx] / (1 - beta1_t)
                v_hat = self.velocities[param_idx] / (1 - beta2_t)
                update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                layer.parameters()[param_idx] -= update
