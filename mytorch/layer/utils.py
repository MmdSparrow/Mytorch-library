from mytorch import Tensor
import numpy as np

def zero_padding(x: Tensor, padding_size=(1,1)):
    x.data = np.pad(x.data, padding_size, mode='constant', constant_values=0)
    return x