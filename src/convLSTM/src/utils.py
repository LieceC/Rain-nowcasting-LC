import numpy as np
import torch
from PIL import Image


def get_date_from_file_name(filename):
    date_infos = [int(val[1:]) for val in filename.split('/')[-1].split('.')[0].split('-')]
    return date_infos


def filter_one_week_over_two_for_eval(idx):
    samples_per_week = 12 * 24 * 7
    return (idx // samples_per_week) % 2


def missing_file_in_sequence(files_names):
    for k in range(len(files_names) - 1):
        month_1, day_1, hour_1, min_1 = get_date_from_file_name(files_names[k])[1:]
        month_2, day_2, hour_2, min_2 = get_date_from_file_name(files_names[k + 1])[1:]

        if (min_1 + 5) % 60 != min_2:
            # print("Min gap : ", files_names, "\n")
            return True
        if (hour_1 + 1) % 24 != hour_2 and (min_1 == 55 and min_2 == 0):
            # print("Hour gap : ", files_names, "\n")
            return True
        if day_1 != day_2 and day_1 + 1 != day_2 and not (
                (day_1 == 30 and day_2 == 1) or (day_1 == 31 and day_2 == 1) or (
                month_1 == 2 and (day_1 == 28 or day_1 == 29) and day_2 == 1)):
            # print("Day gap : ", files_names, "\n")
            return True

    return False


def save_gif(single_seq, fname):
    """Save a single gif consisting of image sequence in single_seq to fname."""
    img_seq = [Image.fromarray(img.astype(np.float32) * 255, 'F').convert("L") for img in single_seq]
    img = img_seq[0]
    img.save(fname, save_all=True, append_images=img_seq[1:])


def save_gifs(seq, prefix):
    """Save several gifs.
    Args:
      seq: Shape (num_gifs, IMG_SIZE, IMG_SIZE)
      prefix: prefix-idx.gif will be the final filename.
    """

    for idx, single_seq in enumerate(seq):
        save_gif(single_seq, "{}-{}.gif".format(prefix, idx))


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
