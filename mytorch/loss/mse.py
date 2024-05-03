from mytorch import Tensor


def MeanSquaredError(preds: Tensor, actual: Tensor):
    "TODO: implement Mean Squared Error loss"
    mse = preds.__sub__(actual).__pow__(2)
    elements_number = preds.shape[0] if preds.shape else 1
    # elements_number = preds.shape[0] if preds.shape and preds.shape[0] else 1
    return Tensor(mse.data / elements_number, mse.requires_grad, mse.depends_on)
