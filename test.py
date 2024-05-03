from mytorch import Tensor
import numpy as np

class AvgPool2d:
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)) -> None:
        super()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        channel_numbers, H, W = x.shape
        H_out = (H + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        data = None
        print(f'data before padding :\n{x.data}')
        if self.padding is not None or self.padding!=(0,0):
            data= np.pad(x.data, self.padding, mode='constant', constant_values=0)
            print(f'data after padding :\n{data}')
        else:
            data= x.data

        output_data = np.zeros((channel_numbers, H_out, W_out))
        for c in range(channel_numbers):
            for h in range(H_out):
                for w in range(W_out):
                    output_data[c, h, w] = np.mean(data[c, h * self.stride[0] + self.padding[0] : h * self.stride[0] + self.padding[0] + self.kernel_size[0],
                                                    w * self.stride[1] + self.padding[1] : w * self.stride[1] + self.padding[1] + self.kernel_size[1]])
                        
        return Tensor(output_data, x.requires_grad, x.depends_on)



# import torch
# import torch.nn as nn
# torch_tensor = torch.tensor([[[1,3,1],[1,1,2]]], dtype=torch.float32)
# avg_pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)
# print(avg_pool(torch_tensor))


# import numpy as np

# # Create an array
# arr = np.array([[1, 2, 3], [1, 1, 1]])

# # Pad the array with zeros using numpy.pad()
# pad_width = (1, 1)  # Pad 2 zeros to the beginning and 3 zeros to the end
# padded_array = np.pad(arr, pad_width, mode='constant', constant_values=0)

# # Print the padded array
# print("Padded Array with Zeros:")
# print(padded_array)



from mytorch import Tensor
import numpy as np

def zero_padding(x: Tensor, padding_size=(1,1)):
    x.data = np.pad(x.data, padding_size, mode='constant', constant_values=0)
    return x

# a= Tensor([[[[1,3,1],[3,1,2]], [[1,3,1],[3,1,2]]], [[[2,2,2],[2,2,2]], [[1,3,1],[3,1,2]]]])
# print(a.data[0])
# print(zero_padding(a).data)
# avgpool= AvgPool2d(1, 2, kernel_size=(2,2), padding=(1,1))
# print(avgpool.forward(a))

# def convolve2d(x, w, stride=1, padding=0):
#     # Calculate the dimensions of the output tensor
#     kernel_height, kernel_width = w.shape
#     n_filters, in_channels, height, width = x.shape
#     out_height = (height + 2*padding - kernel_height) // stride + 1
#     out_width = (width + 2*padding - kernel_width) // stride + 1
#     out_channels = n_filters

#     # Initialize the output tensor
#     output = np.zeros((out_channels, out_height, out_width))

#     # Apply the filter to the input tensor using a sliding window approach
#     for f in range(out_channels):
#         for i in range(out_height):
#             for j in range(out_width):
#                 # Extract the area of the input tensor to convolve with the filter
#                 x_window = x[:, :, stride*i:stride*i+kernel_height, stride*j:stride*j+kernel_width]

#                 # Calculate the sum product of the area and the filter
#                 output[f, i, j] = np.sum(x_window * w[f])

#     return output





class Linear():
    def __init__(self, inputs: int, outputs: int, need_bias: bool = False, mode="xavier") -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.need_bias = need_bias
        self.weight: Tensor = None
        self.bias: Tensor = None
        self.initialize_mode = mode

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"

        print(f'weight:{self.weight}')

        print(f'weight:{x._data}')

        result = x.__matmul__(self.weight)
        if self.need_bias:
            result += self.bias
        return Tensor(result, x.requires_grad, x.depends_on)
    
    def initialize(self):
        "TODO: initialize weight by initializer function (mode)"
        self.weight = Tensor(
            data=initializer((self.inputs, self.outputs), mode=self.initialize_mode),
            requires_grad=True
        )

        "TODO: initialize bias by initializer function (zero mode)"
        if self.need_bias:
            self.bias = Tensor(
                data=initializer((self.inputs, self.outputs), mode='zero'),
                requires_grad=True
            )
    
def xavier_initializer(shape):
    "TODO: implement xavier_initializer" 
    return np.random.randn(*shape) * np.sqrt(1/shape[0], dtype=np.float64)

def he_initializer(shape):
    "TODO: implement he_initializer" 
    return np.random.randn(*shape) * np.sqrt(2/shape[0], dtype=np.float64)

def random_normal_initializer(shape, mean=0.0, stddev=0.05):
    "TODO: implement random_normal_initializer" 
    return np.random.normal(loc=mean, scale=stddev, size=shape)

def zero_initializer(shape):
    "TODO: implement zero_initializer" 
    return np.zeros(shape, dtype=np.float64)

def one_initializer(shape):
    return np.ones(shape, dtype=np.float64)

def initializer(shape, mode="xavier"):
    if mode == "xavier":
        return xavier_initializer(shape)
    elif mode == "he":
        return he_initializer(shape)
    elif mode == "random_normal":
        return random_normal_initializer(shape)
    elif mode == "zero":
        return zero_initializer(shape)
    elif mode == "one":
        return one_initializer(shape)
    else:
        raise NotImplementedError("Not implemented initializer method")
    

# a= Tensor([1,1])
# linear_layer = Linear(2, 2)

# print(linear_layer.forward(a))


def relu(x: Tensor) -> Tensor:
    "TODO: implement relu function"

    # use np.maximum
    data = x.data
    print(data)
    data = np.maximum(data, 0)
    print(data)



# a= Tensor([1,1], requires_grad=True)

# relu(a)


# class myclass:
#     def __init__(self) -> None:
#         pass

#     def test(self, a):
#         outter_test(self, a)

# def outter_test(a):
#     print(a)


# wtf_c = myclass()
# wtf_c.test(2)

base = 10

a= Tensor([100,-10], requires_grad=True)

print(a.data/10)
# print(np.log(a.data)/np.log(base))
