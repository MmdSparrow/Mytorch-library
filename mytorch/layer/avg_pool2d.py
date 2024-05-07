from mytorch import Tensor
from mytorch.layer import Layer
from mytorch import Tensor, Dependency


import numpy as np

class AvgPool2d(Layer):
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

        # if self.padding is not None and self.padding!=(0,0):
        #     batch_data= []
        #     for data in x.data:
        #         channel_data = []
        #         for channel in data:
        #             channel_data.append(zero_padding(channel))
        #         batch_data.append(channel_data)
        #     data= batch_data
        # else:
        #     data= x.data

        data = x.data

        output_data = Tensor(np.zeros((batch_size, channel_numbers, H_out, W_out)))

        for b in range(batch_size):
            for o in range(channel_numbers):
                for i in range(H_out):
                    for j in range(W_out):
                        output_data[b, o, i, j] = (x[b, o, i * self.stride[0] + self.padding[0] : i * self.stride[0] + self.padding[0] + self.kernel_size[0],
                                                        j * self.stride[1] + self.padding[1] : j * self.stride[1] + self.padding[1] + self.kernel_size[1]]).mean()
        
        return output_data
    
    def __str__(self) -> str:
        return "avg pool 2d - kernel: {}, stride: {}, padding: {}".format(self.kernel_size, self.stride, self.padding)


def zero_padding(x, padding_size=(1,1)):
    data= np.pad(x, padding_size, mode='constant', constant_values=0)
    return data