import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, shape, input_size, hidden_size, kernel_size, device, bias=True):
        """

        Initialize a Convolutional LSTM cell.

        :param input_size:
            Number of channels in the input
        :param hidden_size:
            Number of hidden state channels
        :param kernel_size:
            Size of the convolutional kernel use in the cell
        :param bias: bool
            Choose if the cell should contain a bias or not
        """
        super(ConvLSTMCell, self).__init__()

        self.shape = shape
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.bias = bias
        self.device = device
        self.padding = (kernel_size - 1) // 2  # keep same size through in output
        self.bias = bias
        # explain trick
        self.conv = nn.Conv2d(in_channels=self.input_size + self.hidden_size,
                              out_channels=4 * self.hidden_size,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.Wci = nn.Parameter(torch.zeros(1, self.hidden_size, shape[0], shape[1])).to(self.device)
        self.Wcf = nn.Parameter(torch.zeros(1, self.hidden_size, shape[0], shape[1])).to(self.device)
        self.Wco = nn.Parameter(torch.zeros(1, self.hidden_size, shape[0], shape[1])).to(self.device)

    def forward(self, inputs=None, states=None):

        if states is None:
            c = torch.zeros((inputs.size(0), self.hidden_size, self.shape[0],
                             self.shape[1]), dtype=torch.float).to(self.device)
            h = torch.zeros((inputs.size(0), self.hidden_size, self.shape[0],
                             self.shape[1]), dtype=torch.float).to(self.device)
        else:
            h, c = states

        outputs = []
        for index in range(inputs.size(1)):
            # initial inputs
            if inputs is None:
                x = torch.zeros((h.size(0), self.input_size, self.shape[0],
                                 self.shape[1]), dtype=torch.float).to(self.device)
            else:
                x = inputs[:, index, ...]
            cat_x = torch.cat([x, h], dim=1)
            conv_x = self.conv(cat_x)

            i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)

            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            o = torch.sigmoid(o)

            c = f * c + i * torch.tanh(tmp_c)
            h = o * torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs), h, c
