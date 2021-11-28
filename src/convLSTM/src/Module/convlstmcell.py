import torch
import torch.nn as nn


class ConvLSTMCell(nn.module):
    def __init__(self, shape, input_size, hidden_size, kernel_size, bias=True):
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
        self.padding = (kernel_size - 1) // 2  # keep same size through in output
        self.bias = bias
        # explain trick
        self.conv = nn.Conv2d(in_channels=self.input_chans + self.num_features,
                              out_channels=4 * self.hidden_size,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        if hidden_state is None:
            h_cur = torch.zeros(inputs.size(1), self.hidden_size, self.shape[0],
                                self.shape[1]).cuda()
            c_cur = torch.zeros(inputs.size(1), self.hidden_size, self.shape[0],
                                self.shape[1]).cuda()
        else:
            h_cur, c_cur = hidden_state
            output_inner = []
        for i in range(seq_len):
            if inputs is None:
                x = torch.zeros(h_cur.size(0), self.input_channels, self.shape[0],
                                self.shape[1]).cuda()
            else:
                x = inputs[i, ...]
            input_comb = torch.cat([x, h_cur], dim=1)
            cc_i, cc_f, cc_o, cc_g = torch.split(input_comb, self.hidden_dim, dim=1)
            i = torch.sigmoid(cc_i + c_cur * self.Wci)
            f = torch.sigmoid(cc_f + c_cur * self.Wcf)
            c = f * c_cur + i * torch.tanh(cc_g)
            o = torch.sigmoid(cc_o + self.Wco * c)
            h = o * torch.tanh(c)
            output_inner.append(h)

        return h, c, torch.stack(output_inner)

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        if self.Wci is None:
            self.Wci = nn.Parameter(torch.zeros(1, height, width)).cuda()
            self.Wcf = nn.Parameter(torch.zeros(1, height, width)).cuda()
            self.Wco = nn.Parameter(torch.zeros(1, height, width)).cuda()
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
