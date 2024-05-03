from mytorch.optimizer import Optimizer

import numpy as np

"TODO: (optional) implement Adam optimizer"
class Adam(Optimizer):
    def __init__(self, layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.timestep = 0
        self.momentums = {}
        self.velocities = {}
    
    def step(self):
        self.timestep += 1
        for layer in self.layers:
            for param_name, param in layer.parameters.items():
                if param_name not in self.momentums:
                    self.momentums[param_name] = np.zeros_like(param.data)
                    self.velocities[param_name] = np.zeros_like(param.data)

                if param.grad is not None:
                    grad = param.grad.data
                    self.momentums[param_name] = self.beta1 * self.momentums[param_name] + (1 - self.beta1) * grad
                    self.velocities[param_name] = self.beta2 * self.velocities[param_name] + (1 - self.beta2) * grad**2

                    m_hat = self.momentums[param_name] / (1 - self.beta1**self.timestep)
                    v_hat = self.velocities[param_name] / (1 - self.beta2**self.timestep)

                    param.data -= (self.learning_rate * m_hat) / (np.sqrt(v_hat) + self.epsilon)
