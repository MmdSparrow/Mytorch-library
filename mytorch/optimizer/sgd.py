from typing import List
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
            print("########################## layerrrrrrrrrrrrrrrrrr ###########################")
            if layer.weight is not None and layer.weight.requires_grad:
                layer.weight.data = layer.weight.data.__sub__(self.learning_rate * (layer.weight.grad.data))
            print("now go bias........................")
            if layer.need_bias and layer.bias is not None and layer.bias.requires_grad:
                layer.bias.data = layer.bias.data.__sub__(self.learning_rate * (layer.bias.grad.data))
