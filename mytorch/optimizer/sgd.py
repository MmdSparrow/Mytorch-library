from typing import List

from torch import Tensor
from mytorch.layer import Layer
from mytorch.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, layers:List[Layer], learning_rate=0.1):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        "TODO: implement SGD algorithm"
        # for layer in self.layers:
        #     if layer.weight is not None and layer.weight.requires_grad:
        #         print(f'layer grad: {layer.weight.grad}')
        for layer in self.layers:
            # print("########################## layerrrrrrrrrrrrrrrrrr ###########################")
            # print(f'self.learning rate:{self.learning_rate}')
            if layer.weight is not None and layer.weight.requires_grad:
                # print(f'layer.weight.data: {layer.weight.data}')
                # print(f'layer.weight.grad.__mul__(Tensor([self.learning_rate])): {layer.weight.grad.__mul__(Tensor([self.learning_rate]))}')
                layer.weight = layer.weight.__sub__(layer.weight.grad.__mul__(Tensor([self.learning_rate])))
                # print(f'after update layer.weight.data: {layer.weight.data}')

            if layer.need_bias and layer.bias is not None and layer.bias.requires_grad:
                # print("in bias")
                layer.bias = layer.bias.__sub__(layer.bias.grad.__mul__(Tensor([self.learning_rate])))
