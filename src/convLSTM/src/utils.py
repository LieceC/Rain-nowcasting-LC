import os
import torch

def get_date_from_file_name(filename):

    date_infos = [int(val[1:]) for val in filename.split('/')[-1].split('.')[0].split('-')]
    return date_infos


def weighted_mse_loss(input, target):

    threshold = [0, 2, 5, 10, 30, 1000]
    weights = [1, 2, 5, 10, 30]
    assert len(threshold) == len(weights) + 1
    loss = torch.Tensor([0])

    for k in range(len(weights)):
        mask = ((threshold[k] <= target) & (target < threshold[k+1])).float()
        loss += torch.sum(weights[k] * ((input*mask - target*mask)) ** 2)

    return loss


def tensorify(lst):
    """
    List must be nested list of tensors (with no varying lengths within a dimension).
    Nested list of nested lengths [D1, D2, ... DN] -> tensor([D1, D2, ..., DN)

    :return: nested list D
    """
    # base case, if the current list is not nested anymore, make it into tensor
    if type(lst[0]) != list:
        if type(lst) == torch.Tensor:
            return lst
        elif type(lst[0]) == torch.Tensor:
            return torch.stack(lst, dim=0)
        else:  # if the elements of lst are floats or something like that
            return torch.tensor(lst)
    current_dimension_i = len(lst)
    for d_i in range(current_dimension_i):
        tensor = tensorify(lst[d_i])
        lst[d_i] = tensor
    # end of loop lst[d_i] = tensor([D_i, ... D_0])
    tensor_lst = torch.stack(lst, dim=0)
    return tensor_lst