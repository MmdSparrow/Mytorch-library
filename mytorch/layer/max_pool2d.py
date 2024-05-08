from mytorch import Tensor
from mytorch.layer import Layer

import numpy as np

class MaxPool2d(Layer):
    def __init__(self, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)) -> None:
        super()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        batch_size, channel_numbers, H, W = x.shape
        H_out = (H + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        if self.padding is not None and self.padding!=(0,0):
            x=x.zero_padding(self.padding)

        output = Tensor(np.zeros((batch_size, channel_numbers, H_out, W_out)))

        for n in range(batch_size):
            for c in range(channel_numbers):
                for h in range(H_out):
                    for w in range(W_out):
                        output[n, c, h, w] = (x[n, c, h * self.stride[0] + self.padding[0] : h * self.stride[0] + self.padding[0] + self.kernel_size[0],
                                                        w * self.stride[1] + self.padding[1] : w * self.stride[1] + self.padding[1] + self.kernel_size[1]]).max()
        
        return output
    
    def __str__(self) -> str:
        return "max pool 2d - kernel: {}, stride: {}, padding: {}".format(self.kernel_size, self.stride, self.padding)

def zero_padding(x, padding_size=(1,1)):
    data= np.pad(x, padding_size, mode='constant', constant_values=0)
    return data