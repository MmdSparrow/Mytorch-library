from mytorch import Tensor


def MeanSquaredError(preds: Tensor, actual: Tensor):
    "TODO: implement Mean Squared Error loss"
    squared_diff = (preds.__sub__(actual)).__pow__(2)

    result = squared_diff.sum()

    num_elements = preds.shape[0] if preds.shape else 1
    result.data = result.data / num_elements

    return result