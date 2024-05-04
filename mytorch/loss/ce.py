import numpy as np
from mytorch import Tensor

def CategoricalCrossEntropy(preds: Tensor, label: Tensor):
    "TODO: implement Categorical Cross Entropy loss"
    label_array= np.zeros_like(preds.data)
    for i in range(preds.shape[0]):
        label_array[i][int(label.data[i])]=1
    p_start = Tensor(label_array)
    # result = (preds.log().__mul__(label).sum()).__neg__()
    result = (preds.log().__mul__(p_start).sum()).__neg__()
    # print(f'result shape: {result.shape}')
    return result