from src.Module.convlstmcell import *

"""
Implémentation of the ConvLSTM architecture from this paper : https://arxiv.org/pdf/1506.04214.pdf
"""
trainFolder = None
validFolder = None
trainLoader = torch.utils.data.DataLoader()
validLoader = torch.utils.data.DataLoader()


class ConvLSTM(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=True):
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.encoder = nn.sequential(ConvLSTMCell(input_dim=input_dim,
                                                  hidden_dim=self.hidden_dim,
                                                  kernel_size=self.kernel_size,
                                                  bias=bias),
                                     ConvLSTMCell(input_dim=input_dim,
                                                  hidden_dim=self.hidden_dim,
                                                  kernel_size=self.kernel_size,
                                                  bias=bias))
        self.decoder = nn.sequential(ConvLSTMCell(input_dim=input_dim,
                                                  hidden_dim=self.hidden_dim,
                                                  kernel_size=self.kernel_size,
                                                  bias=bias),
                                     ConvLSTMCell(input_dim=input_dim,
                                                  hidden_dim=self.hidden_dim,
                                                  kernel_size=self.kernel_size,
                                                  bias=bias)
                                     )

    def forward(self, input):
        state = self.encoder(input)
        output = self.decoder(state)
        return output


def train(data):
    model = ConvLSTM(input_shape=(512, 512), input_dim=3, hidden_dim=64, kernel_size=3)
