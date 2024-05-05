from torch import Tensor
from mytorch.optimizer import Optimizer
import numpy as np

class Momentum(Optimizer):
    def __init__(self, layers, learning_rate=0.1, momentum=0.9):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}
        for layer_index, layer in enumerate(self.layers):
            self.velocities[(layer_index, "w")] = Tensor(np.zeros_like(layer.weight.data))
            if layer.need_bias:
                self.velocities[(layer_index, "b")] = Tensor(np.zeros_like(layer.bias.data))

    def step(self):
        layer_index= -1
        for layer in self.layers:
            layer_index+=1
            if layer.weight is not None and layer.weight.requires_grad:
                # if (layer_index, param_index) not in self.velocities:
                    # self.velocities[(layer_index, param_index)] = Tensor(np.zeros_like(param.data))

                if layer.weight.grad is not None:
                    self.velocities[(layer_index, "w")] = (
                        # self.velocities[(layer_index, param_index, "w")].__mul__(Tensor([self.momentum])).__add__(Tensor([self.learning_rate]).__mul__(param.grad))
                        self.momentum * self.velocities[(layer_index, "w")] +
                        self.learning_rate * layer.weight.grad
                    )
                    layer.weight = layer.weight - self.velocities[(layer_index, "w")]
                        
            if layer.need_bias and layer.bias is not None and layer.bias.requires_grad:
                # if (layer_index, param_index) not in self.velocities:
                    # self.velocities[(layer_index, param_index)] = Tensor(np.zeros_like(param.data))

                if layer.bias.grad is not None:
                    self.velocities[(layer_index, "b")] = (
                            self.momentum * self.velocities[(layer_index, "b")] +
                            self.learning_rate * layer.bias.grad
                    )
                    layer.bias = layer.bias - self.velocities[(layer_index, "b")]
            
            
