import numpy as np
from mytorch import Tensor, Dependency

def tanh(x: Tensor) -> Tensor:
    """
    TODO: (optional) implement tanh function
    hint: you can do it using function you've implemented (not directly define grad func)
    """
    ex = x.exp()
    ex_neg = (x.__neg__()).exp()
    numerator = ex.__sub__(ex_neg)
    denumerator = ex.__add__(ex_neg)
    return numerator.__mul__(denumerator.__pow__(-1))