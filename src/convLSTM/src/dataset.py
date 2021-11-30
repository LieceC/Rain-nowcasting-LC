import os

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import get_date_from_file_name


class MeteoDataset(Dataset):

    def __init__(self, rain_dir, input_length, output_length):

        self.rain_dir = rain_dir
        self.input_length = input_length
        self.output_length = output_length
        self.n_frames_total = self.input_length + self.output_length
        self.files_names = [f for f in os.listdir(rain_dir)[:50] if os.path.isfile(os.path.join(rain_dir, f))]
        self.files_names = sorted(self.files_names, key=lambda x: get_date_from_file_name(x))
        # print(self.files_names)

    def __len__(self):

        return len(self.files_names) - self.n_frames_total

    def __getitem__(self, i):

        files_names_i = self.files_names[i: i + self.input_length + self.output_length]
        path_files = [os.path.join(self.rain_dir, file_name) for file_name in files_names_i]

        # Create a sequence of input rain maps.
        rain_map = np.load(path_files[0])['arr_0']
        rain_map = torch.unsqueeze(torch.from_numpy(rain_map).float(), dim=0)[None, :]
        rain_sequence_data = rain_map

        for k in range(1, self.input_length):
            rain_map = np.load(path_files[k])['arr_0']
            rain_map = torch.unsqueeze(torch.from_numpy(rain_map).float(), dim=0)[None, :]
            rain_sequence_data = torch.cat((rain_sequence_data, rain_map), dim=0)

        # Create a sequence of target rain maps.
        rain_map = np.load(path_files[self.output_length])['arr_0']
        rain_map = torch.unsqueeze(torch.from_numpy(rain_map).float(), dim=0)[None, :]
        rain_sequence_target = rain_map
        for k in range(self.input_length + 1, self.output_length + self.input_length):
            rain_map = np.load(path_files[k])['arr_0']
            rain_map = torch.unsqueeze(torch.from_numpy(rain_map).float(), dim=0)[None, :]
            rain_sequence_target = torch.cat((rain_sequence_target, rain_map), 0)

        return {"input": rain_sequence_data, "target": rain_sequence_target}
