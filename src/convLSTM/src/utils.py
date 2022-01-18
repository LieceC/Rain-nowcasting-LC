import os
import glob
import numpy as np
import torch
from PIL import Image

def save_gif_2(single_seq, fname):
    if len(single_seq.shape) == 4:
        single_seq = single_seq.permute((0, 2, 3, 1))
        single_seq = torch.squeeze(single_seq, -1)
    single_seq = single_seq.cpu().detach().numpy()
    single_seq_masked = rain_map_thresholded(single_seq)
    img_seq = [Image.fromarray((img*255).astype(np.uint8), 'RGB') for img in single_seq_masked]
    img = img_seq[0]
    img.save(fname, save_all=True, append_images=img_seq[1:], duration=500, loop=0)


def rain_map_thresholded(single_seq):
    single_seq_masked = np.zeros((single_seq.shape[0], single_seq.shape[1], single_seq.shape[2], 3))
    single_seq_masked[:, :, :, 1] = np.where((0.1 < single_seq) & (single_seq < 1.), 1., 0.)
    single_seq_masked[:, :, :, 2] = np.where((1. < single_seq) & (single_seq < 2.5), 1., 0.)
    single_seq_masked[:, :, :, 0] = np.where((2.5 < single_seq), 1., 0.)
    return single_seq_masked

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

def create_dir(dir):
  if not os.path.exists(dir):
    os.makedirs(dir)
    print("Created Directory : ", dir)
  else:
    files = glob.glob(dir+'/*')
    for f in files:
      os.remove(f)
  return dir

def save_gif(single_seq_pred, single_seq_gt, fname):
    """Save a single gif consisting of image sequence in single_seq to fname."""
    create_dir(fname)
    # [S,I,H,W]
    single_seq = torch.permute(single_seq_pred, (0, 2, 3, 1))
    single_seq = torch.squeeze(single_seq, -1)
    single_seq = single_seq.cpu().detach().numpy()
    img_seq = [Image.fromarray(img.astype(np.float32) * 255, 'F').convert("L") for img in single_seq]
    img = img_seq[0]
    img.save(fname+"/pred.gif", save_all=True, append_images=img_seq[1:])

    single_seq = torch.permute(single_seq_gt, (0, 2, 3, 1))
    single_seq = torch.squeeze(single_seq, -1)
    single_seq = single_seq.cpu().detach().numpy()
    img_seq = [Image.fromarray(img.astype(np.float32) * 255, 'F').convert("L") for img in single_seq]
    img = img_seq[0]
    img.save(fname+"/gt.gif", save_all=True, append_images=img_seq[1:])


def save_gifs(seqpred, seqgt, prefix):
    """Save several gifs.
    Args:
      seq: Shape (num_gifs, IMG_SIZE, IMG_SIZE)
      prefix: prefix-idx.gif will be the final filename.
    """
    for idx, (single_seq_pred, single_seq_gt) in enumerate(zip(seqpred, seqgt)):
        save_gif(single_seq_pred, single_seq_gt, "{}-{}".format(prefix, idx))


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


def weighted_mse_loss(output, target, weight_mask):
    return torch.sum(torch.multiply(weight_mask, (output - target) ** 2))


def weighted_mae_loss(output, target, weight_mask):
    return torch.sum(torch.multiply(weight_mask, torch.abs(output - target)))


def compute_weight_mask(target):
    """
    threshold = [0, 2, 5, 10, 30, 1000]
    weights = [1., 2., 5., 10., 30.]
    # To solve
    mask = torch.ones(target.size(), dtype=torch.double).cuda()
    for k in range(len(weights)):
        mask = torch.where((threshold[k] <= target) & (target < threshold[k+1]), weights[k], mask)
    """

    ### Fix for small gpu below
    return torch.where((0 <= target) & (target < 0.1), 1., 0.) \
           + torch.where((0.1 <= target) & (target < 1), 1.2, 0.) \
           + torch.where((1 <= target) & (target < 2.5), 1.5, 0.) \
           + torch.where((2.5 <= target), 2., 0.)