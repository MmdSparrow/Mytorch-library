import numpy as np
from mytorch import Tensor, Dependency

def leaky_relu(x: Tensor) -> Tensor:
    """
    TODO: implement leaky_relu function.
    fill 'data' and 'req_grad' and implement LeakyRelu grad_fn
    hint: use np.where like Relu method but for LeakyRelu
    """
    # we suppose that leak coefficient or alpha is equal to 0.01
    ALPHA_CONSTANT = 0.01
    data = np.where(x._data > 0, x._data, ALPHA_CONSTANT* x._data)
    req_grad = x.requires_grad

    if req_grad:
        def grad_fn(grad: np.ndarray):
            return grad * np.where(x._data > 0, grad, ALPHA_CONSTANT * grad)

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)


