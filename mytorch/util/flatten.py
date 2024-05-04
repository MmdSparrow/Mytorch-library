from mytorch import Tensor

def flatten(x: Tensor) -> Tensor:
    """
    TODO: implement flatten. 
    this methods transforms a n dimensional array into a flat array
    hint: use numpy flatten
    """
    batch_data= []
    for data in x._data:
        batch_data.append(data.flatten())
    data= batch_data
    
    req_grad = x.requires_grad
    depends_on = x.depends_on
    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)


    