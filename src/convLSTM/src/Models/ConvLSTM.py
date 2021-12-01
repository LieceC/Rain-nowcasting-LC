from src.Module.convlstmcell import *

"""
Implémentation of the ConvLSTM architecture from this paper : https://arxiv.org/pdf/1506.04214.pdf
"""


class ConvLSTM(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTM, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.nb_layers = 4
        self.lstmcell1 = ConvLSTMCell(shape=input_shape,
                                      input_size=input_dim,
                                      hidden_size=self.hidden_dim,
                                      kernel_size=self.kernel_size,
                                      bias=bias)
        self.lstmcell2 = ConvLSTMCell(shape=input_shape,
                                      input_size=hidden_dim,
                                      hidden_size=self.hidden_dim,
                                      kernel_size=self.kernel_size,
                                      bias=bias)
        self.lstmcell3 = ConvLSTMCell(shape=input_shape,
                                      input_size=hidden_dim,
                                      hidden_size=self.hidden_dim,
                                      kernel_size=self.kernel_size,
                                      bias=bias)
        self.lstmcell4 = ConvLSTMCell(shape=input_shape,
                                      input_size=hidden_dim,
                                      hidden_size=self.input_dim,
                                      kernel_size=self.kernel_size,
                                      bias=bias)
        self.cell_list = nn.ModuleList([self.lstmcell1, self.lstmcell2, self.lstmcell3, self.lstmcell4])

    def forward(self, input, hidden_state=None):
        hidden_state = self._init_hidden(batch_size=input.shape[0],
                                         image_size=self.input_shape)
        layer_output_list = []
        last_state_list = []

        seq_len = input.size(1)
        cur_layer_input = input

        for layer_idx in range(self.nb_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](inputs=cur_layer_input[:, t, :, :, :],
                                                 hidden_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        return [self.lstmcell1.init_hidden(batch_size, image_size),
                self.lstmcell2.init_hidden(batch_size, image_size),
                self.lstmcell3.init_hidden(batch_size, image_size),
                self.lstmcell4.init_hidden(batch_size, image_size)]
