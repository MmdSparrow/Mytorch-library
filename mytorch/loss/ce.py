import numpy as np
from mytorch import Tensor

def CategoricalCrossEntropy(preds: Tensor, label: Tensor):
    "TODO: implement Categorical Cross Entropy loss"
    if np.isnan(preds.data).any() or np.any(preds.data < 0):
        raise ValueError("invalid value in data!")
    label_array= np.zeros_like(preds.data)
    for i in range(preds.shape[0]):
        label_array[i][int(label.data[i])]=1
    p_star = Tensor(label_array)
    result = (p_star.__mul__(preds.log())).sum().__neg__()
    return result



    # return ((label * (preds.log())) + (1 - label) * ((1 - preds).log())).sum()