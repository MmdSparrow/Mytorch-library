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

    def step(self):
        for l in self.layers:
            for param_name, param in l.parameters.items():
                if param_name not in self.accumulators:
                    self.accumulators[param_name] = np.zeros_like(param.data)

                if param.grad is not None:
                    grad = param.grad.data
                    self.accumulators[param_name] = self.rho * self.accumulators[param_name] + (1 - self.rho) * grad**2
                    param.data -= (self.learning_rate * grad) / (np.sqrt(self.accumulators[param_name]) + self.epsilon)
