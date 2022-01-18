import torch
import torch.nn.functional as F
from torch import nn

from src.Models.convlstmcell import ConvLSTMCell

"""
Implémentation of the ConvLSTM architecture from this paper : https://arxiv.org/pdf/1506.04214.pdf
"""


class ConvLSTM(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, device, bias=True):
        super(ConvLSTM, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.device = device
        self.nb_layers = 4
        self.encoding1 = ConvLSTMCell(shape=input_shape,
                                      input_size=input_dim,
                                      hidden_size=self.hidden_dim,
                                      kernel_size=self.kernel_size,
                                      device=device,
                                      bias=bias)
        self.encoding2 = ConvLSTMCell(shape=input_shape,
                                      input_size=hidden_dim,
                                      hidden_size=self.hidden_dim,
                                      kernel_size=self.kernel_size,
                                      device=device,
                                      bias=bias)
        self.decoding1 = ConvLSTMCell(shape=input_shape,
                                      input_size=hidden_dim,
                                      hidden_size=self.hidden_dim,
                                      kernel_size=self.kernel_size,
                                      device=device,
                                      bias=bias)
        self.decoding2 = ConvLSTMCell(shape=input_shape,
                                      input_size=hidden_dim,
                                      hidden_size=self.hidden_dim,
                                      kernel_size=self.kernel_size,
                                      device=device,
                                      bias=bias)

        self.out_conv1 = nn.Conv3d(in_channels=self.hidden_dim,
                                   out_channels=self.hidden_dim // 4,
                                   kernel_size=1,
                                   padding=0,
                                   bias=bias)
        self.out_conv2 = nn.Conv3d(in_channels=self.hidden_dim // 4,
                                   out_channels=self.input_dim,
                                   kernel_size=1,
                                   padding=0,
                                   bias=bias)

    def forward(self, input):
        outputs, h1, c1 = self.encoding1(inputs=input, states=None)
        outputs, h2, c2 = self.encoding2(inputs=outputs, states=None)
        outputs, _, _ = self.decoding1(inputs=outputs, states=(h1, c1))
        outputs, _, _ = self.decoding2(inputs=outputs, states=(h2, c2))

        layer_output_list = outputs
        layer_output_list = torch.permute(layer_output_list, (0, 2, 1, 3, 4))
        layer_output_list = F.relu(self.out_conv1(layer_output_list))
        layer_output_list = F.relu(self.out_conv2(layer_output_list))
        layer_output_list = torch.permute(layer_output_list, (0, 2, 1, 3, 4))
        return layer_output_list

