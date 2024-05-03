from mytorch.optimizer import Optimizer
import numpy as np

"TODO: (optional) implement Momentum optimizer"
class Momentum(Optimizer):
    def __init__(self, layers, learning_rate=0.1, momentum=0.9):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}

    def step(self):
        for layer in self.layers:
            for param_name, param in layer.parameters.items():
                if param_name not in self.velocities:
                    self.velocities[param_name] = np.zeros_like(param.data)

                if param.grad is not None:
                    self.velocities[param_name] = (
                            self.momentum * self.velocities[param_name] +
                            self.learning_rate * param.grad.data
                    )
                    param.data -= self.velocities[param_name]
