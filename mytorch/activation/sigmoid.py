import numpy as np
from mytorch import Tensor, Dependency

def sigmoid(x: Tensor) -> Tensor:
    """
    TODO: implement sigmoid function
    hint: you can do it using function you've implemented (not directly define grad func)
    """
    one_tensor = Tensor(np.ones(x.shape), x.requires_grad, x.depends_on)
    denominator = one_tensor.__add__((x.__neg__()).exp())

    return one_tensor.__mul__(denominator.__pow__(-1))