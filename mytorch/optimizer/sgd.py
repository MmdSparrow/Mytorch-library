from typing import List
import numpy as np

from torch import Tensor
from mytorch.layer import Layer
from mytorch.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, layers:List[Layer], learning_rate=0.1):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        "TODO: implement SGD algorithm"
        for layer in self.layers:
            if layer.weight is not None and layer.weight.requires_grad:
                layer.weight = layer.weight.__sub__(layer.weight.grad.__mul__(Tensor([self.learning_rate])))
            if layer.need_bias and layer.bias is not None and layer.bias.requires_grad:
                layer.bias = layer.bias.__sub__(layer.bias.grad.__mul__(Tensor([self.learning_rate])))


        # for l in self.layers:
        #     l.weight = l.weight - (self.learning_rate * l.weight.grad)
        #     if l.need_bias:
        #         l.bias = l.bias - (self.learning_rate * l.bias.grad)
