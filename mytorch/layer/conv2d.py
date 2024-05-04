from mytorch import Tensor
from mytorch.layer import Layer
from mytorch.util import initializer


import numpy as np

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), need_bias: bool = False, mode="xavier") -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.need_bias = need_bias
        self.weight: Tensor = None
        self.bias: Tensor = None
        self.initialize_mode = mode

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"

        # padding
        if self.padding is not None or self.padding!=(0,0):
            batch_data= []
            for data in x.data:
                channel_data = []
                for channel in data:
                    channel_data.append(zero_padding(channel))
                batch_data.append(channel_data)
            data= batch_data
        else:
            data= x.data

        
        
    
    def initialize(self):
        "TODO: initialize weight by initializer function (mode)"
        self.weight = Tensor(
            data=initializer((self.kernel_size[0], self.kernel_size[1]), mode=self.initialize_mode),
            requires_grad=True,            
        )

        "TODO: initialize bias by initializer function (zero mode)"
        if self.need_bias:
            self.bias = Tensor(
                data=initializer((1, self.outputs), mode='zero'),
                requires_grad=True,
            )

    def zero_grad(self):
        "TODO: implement zero grad"
        self.weight.zero_grad()
        if self.need_bias:
            self.bias.zero_grad()

    def parameters(self):
        "TODO: return weights and bias"
        if self.need_bias:
            return [self.weight, self.bias]
        return [self.weight]
    
    def __str__(self) -> str:
        return "conv 2d - total params: {} - kernel: {}, stride: {}, padding: {}".format(
                                                                                    self.kernel_size[0] * self.kernel_size[1],
                                                                                    self.kernel_size,
                                                                                    self.stride, self.padding)

def zero_padding(x, padding_size=(1,1)):
    data= np.pad(x, padding_size, mode='constant', constant_values=0)
    return data