from ast import List

from torch import tensor
from mytorch import Tensor
import numpy as np

from mytorch.tensor import Tensorable

# class AvgPool2d:
#     def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)) -> None:
#         super()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding

#     def forward(self, x: Tensor) -> Tensor:
#         "TODO: implement forward pass"
#         channel_numbers, H, W = x.shape
#         H_out = (H + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
#         W_out = (W + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

#         data = None
#         print(f'data before padding :\n{x.data}')
#         if self.padding is not None or self.padding!=(0,0):
#             data= np.pad(x.data, self.padding, mode='constant', constant_values=0)
#             print(f'data after padding :\n{data}')
#         else:
#             data= x.data

#         output_data = np.zeros((channel_numbers, H_out, W_out))
#         for c in range(channel_numbers):
#             for h in range(H_out):
#                 for w in range(W_out):
#                     output_data[c, h, w] = np.mean(data[c, h * self.stride[0] + self.padding[0] : h * self.stride[0] + self.padding[0] + self.kernel_size[0],
#                                                     w * self.stride[1] + self.padding[1] : w * self.stride[1] + self.padding[1] + self.kernel_size[1]])
                        
#         return Tensor(output_data, x.requires_grad, x.depends_on)



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

# def zero_padding(x: Tensor, padding_size=(1,1)):
    # x.data = np.pad(x.data, padding_size, mode='constant', constant_values=0)
    # return x

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

# a= Tensor([100,-10], requires_grad=True)
# # print(a.shape)

# # print(a.data/10)
# # print(np.log(a.data)/np.log(base))




# def zero_padding(x: List, padding_size=(1,1)):
#     data= np.pad(x, padding_size, mode='constant', constant_values=0)
#     return data

# a= Tensor([[[[1,3,1],[3,1,2]], [[1,3,1],[3,1,2]]], [[[2,2,2],[2,2,2]], [[1,3,1],[3,1,2]]]])
# print(a.data)
# print('################################')
# # # print(zero_padding(a))

# # # Vectorize the function
# # do_zero_padding = np.vectorize(zero_padding)

# # # Apply the vectorized function along the rows (axis=1)
# # result = np.apply_along_axis(do_zero_padding, axis=1, arr=a.data)
# # print(result)

# batch_data=[]

# for data in a.data:
#     channel_data = []
#     for channel in data:
#         channel_data.append(zero_padding(channel))
#     batch_data.append(channel_data)

# print(Tensor(batch_data).data)

# def CategoricalCrossEntropy(preds: Tensor, label: Tensor):
#     "TODO: implement Categorical Cross Entropy loss"
#     label_array= np.zeros_like(preds.data)
#     label_array[int(label.data)]=1
#     p_start = Tensor(label_array)
#     # result = (preds.log().__mul__(label).sum()).__neg__()
#     result = (preds.log().__mul__(p_start).sum()).__neg__()
#     print(f'result shape: {result.shape}')
#     return result

# pred = Tensor(np.array([0.2,0.7,0.1]))
# scaler = Tensor(np.array([2]))
# print(pred.__mul__(scaler))
# label = Tensor(np.array([1]))
# print(pred)
# print(label)
# print(CategoricalCrossEntropy(pred, label))


# Tensor(round(1/self.learning_rate,3))


# def flatten(x) -> Tensor:
#     """
#     TODO: implement flatten. 
#     this methods transforms a n dimensional array into a flat array
#     hint: use numpy flatten
#     """
#     print(f'this is x: {x}')
#     return x.flatten()


# a= Tensor([[[1,3,1],[3,1,2]], [[1,3,1],[3,1,2]]])
# print(a.data)


# do_zero_padding = np.vectorize(flatten)

# # Apply the vectorized function along the rows (axis=1)
# result = np.apply_along_axis(do_zero_padding, axis=0, arr=a.data)
# print(result)

# Tensor([[-0.04155506 -1.09098493  0.26597207]
#  [ 0.88602766  0.33213968  0.33549567]
#  [-0.93638785 -0.3665651   0.06151861]
#  [ 0.41464566  0.69636617 -0.96624027]
#  [ 0.01372388 -0.04445313  0.38072196]
#  [-0.4551334  -0.40155886 -0.92272375]
#  [ 1.24411352 -1.22645905 -1.19784186]
#  [-2.2295188  -0.39371019  0.29233979]
#  [-1.39699156 -1.26996797  1.2679163 ]
#  [ 1.32229145  1.46435377 -0.1158694 ]], requires_grad=False)



###################################################################################################
# inputs= np.array(
# [[-0.04155506, -1.09098493 , 0.26597207]
#  ,[ 0.88602766, 0.33213968, 0.33549567]]
#  )

# weight = np.array(
# [[ 0.83757601],
#  [-1.34889435],
#  [ 0.4034029 ]]
#  )

# output = np.array(
# [[4.56671604],
#  [5.77683151]]
# )

# preds = inputs @ weight


# actual = output
# loss=((((preds[0] - actual[0])*(preds[0] - actual[0]))+((preds[1] - actual[1])*(preds[1] - actual[1])))/2)
# print(f'loss: {loss}')

# cost_per_y_round=((((preds[0] - actual[0]))+((preds[1] - actual[1]))))
# print(cost_per_y_round)

# total_gradient_1 = weight[0][0] * cost_per_y_round
# total_gradient_2 = weight[1][0] * cost_per_y_round
# total_gradient_3 = weight[2][0] * cost_per_y_round


# total_gradient_1 = total_gradient_1 * 0.01
# total_gradient_2 = total_gradient_2 * 0.01
# total_gradient_3 = total_gradient_3 * 0.01


# new_w_1 = weight[0][0] - total_gradient_1
# new_w_2 = weight[1][0] - total_gradient_1
# new_w_3 = weight[2][0] - total_gradient_1
# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11")
# print(new_w_1)
# print(new_w_2)
# print(new_w_3)


#########################################################################################################
# test soft max

# def softmax(x: Tensor) -> Tensor:
#     """
#     TODO: implement softmax function
#     hint: you can do it using function you've implemented (not directly define grad func)
#     hint: you can't use sum because it has not axis argument so there are 2 ways:
#         1. implement sum by axis
#         2. using matrix mul to do it :) (recommended)
#     hint: a/b = a*(b^-1)
#     """
#     # SM = self.value.reshape((-1,1))
#     # jac = np.diagflat(self.value) - np.dot(SM, SM.T)
#     exp = x.exp()
#     denominator = exp.__matmul__(np.ones((exp.shape[-1], 1)))
#     result = exp.__mul__(denominator.__pow__(-1))
#     return result

# a = Tensor([[-1, 0, 3, 5], [-1, 0, 3, 5]])
# b = Tensor([[-1, 0, 3, 5]])
# print(softmax(a))
# print(softmax(b))


#########################################################################################################

# def CategoricalCrossEntropy(preds: Tensor, label: Tensor):
#     "TODO: implement Categorical Cross Entropy loss"
#     label_array= np.zeros_like(preds.data)
#     for i in range(preds.shape[0]):
#         label_array[i][int(label.data[i])]=1
#     p_start = Tensor(label_array)
#     # result = (preds.log().__mul__(label).sum()).__neg__()
#     result = (preds.log().__mul__(p_start).sum()).__neg__()
#     # print(f'result shape: {result.shape}')
#     return result

# preds = Tensor([[0.2, 0.7, 0.1],[0.8, 0.1, 0.1],[0.1, 0.1, 0.9]])
# label = Tensor([1,0,2])

# print(CategoricalCrossEntropy(preds, label))

#####################################################################################################################

# a= Tensor([[1,2,3],[1,2,3],[1,2,3],[1,2,3]])
# b= Tensor([1,1,1])

# print(a.__add__(b))

# yp= np.array([9, 4, 4, 8, 4, 4, 4, 4, 4, 6, 4, 4, 4, 4, 8, 9, 4, 4, 4, 4, 4, 8,
#        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 4, 9, 4, 4, 8, 4, 9,
#        4, 4, 4, 4, 6, 4, 9, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 4, 4])

# label= np.array([4, 0, 0, 1, 7, 3, 6, 8, 4, 1, 6, 0, 0, 2, 1, 1, 5, 5, 8, 0, 5, 9,
#        6, 4, 5, 9, 2, 7, 7, 8, 2, 3, 9, 4, 3, 9, 1, 7, 7, 2, 4, 1, 3, 5,
#        5, 6, 4, 4, 2, 6, 6, 7, 5, 4, 7, 0, 9, 6, 9, 2, 1, 4, 6, 3])


# print(np.sum(yp==label))


# a= np.array([0.09984804, 0.10061626, 0.10487659, 0.10884891, 0.08314399,
#        0.09931026, 0.0912681 , 0.10296731, 0.1165465 , 0.09257403])

# print(a.sum())


# #################################

# a = Tensor([[1,2,3],[1,1,1],[0.5, 1, 3]])
# b = Tensor([[2,2,2],[3,3,3],[1, 1, 1]])

# print(a.__mul__(b).sum())




# class AvgPool2d():
#     def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)) -> None:
#         super()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding

#     def forward(self, x: Tensor) -> Tensor:

#         def zero_padding(x, padding_size=(1,1)):
#             data= np.pad(x, padding_size, mode='constant', constant_values=0)
#             return data
        
#         "TODO: implement forward pass"
#         batch_size, channel_numbers, H, W = x.shape
#         H_out = (H + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
#         W_out = (W + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

#         if self.padding is not None or self.padding!=(0,0):
#             batch_data= []
#             for data in x.data:
#                 channel_data = []
#                 for channel in data:
#                     channel_data.append(zero_padding(channel))
#                 batch_data.append(channel_data)
#             data= batch_data
#         else:
#             data= x.data

#         output_data = np.zeros((batch_size, channel_numbers, H_out, W_out))
#         for n in range(batch_size):
#             for c in range(channel_numbers):
#                 for h in range(H_out):
#                     for w in range(W_out):
#                         output_data[n, c, h, w] = np.mean(data[n, c, h * self.stride[0] + self.padding[0] : h * self.stride[0] + self.padding[0] + self.kernel_size[0],
#                                                         w * self.stride[1] + self.padding[1] : w * self.stride[1] + self.padding[1] + self.kernel_size[1]])
        
#         return Tensor(output_data, x.requires_grad, x.depends_on)
    
#     def __str__(self) -> str:
#         return "avg pool 2d - kernel: {}, stride: {}, padding: {}".format(self.kernel_size, self.stride, self.padding)
    
# a= Tensor([[[[1,3,1],[3,1,2]], [[1,3,1],[3,1,2]]], [[[2,2,2],[2,2,2]], [[1,3,1],[3,1,2]]]])

# my_average_pool= AvgPool2d()
# print(my_average_pool.forward(a))

# a= Tensor([[[[1,3,1],[3,1,2]], [[1,3,1],[3,1,2]]], [[[2,2,2],[2,2,2]], [[1,3,1],[3,1,2]]]])

# # a.__getitem__(0).__getitem__(0).__setitem__(0, Tensor([[0,0,0],[0,0,0]]))
# print(a[0, 0])


a= Tensor([[[[1,3,1],[3,1,2]], [[-1,3,-1],[3,1,2]]], [[[2,-2,-2],[2,2,2]], [[-1,-3,1],[3,1,-2]]]])
print(Tensor(np.where(a.data> 0, 1, 0)))