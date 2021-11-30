import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.Models.ConvLSTM import ConvLSTM, torch
from src.dataset import MeteoDataset
from src.utils import tensorify


def trainer(input_shape=(128, 128), input_dim=1, hidden_dim=64, kernel_size=3, input_length=1, output_length=1,
            batch_size=20, epochs=50):
    board = SummaryWriter()
    net = ConvLSTM(input_shape=input_shape,
                   input_dim=input_dim,
                   hidden_dim=hidden_dim,
                   kernel_size=kernel_size)

    train = MeteoDataset(rain_dir='C:\\Users\\liece\\MeteoNet-Brest\\rainmap\\train', input_length=input_length,
                         output_length=output_length)
    val = MeteoDataset(rain_dir='C:\\Users\\liece\\MeteoNet-Brest\\rainmap\\val', input_length=input_length,
                       output_length=output_length)

    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=False)
    valid_dataloader = DataLoader(val, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(net.parameters())
    loss_f = torch.nn.MSELoss()

    device = torch.device("cpu")

    net.to(device)

    avg_train_losses = []
    avg_val_losses = []

    # set checkpoint
    cur_epoch = 0
    for epoch in range(cur_epoch, epochs):
        train_losses = []
        val_losses = []
        net.train()
        t = tqdm(train_dataloader, leave=False, total=len(train_dataloader))
        for batch_idx, sample in enumerate(t):
            inputs, targets = sample['input'], sample['target']
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            pred = tensorify(net(inputs)[0])
            pred = torch.squeeze(pred, 0)
            loss = loss_f(pred, targets)
            average_loss = loss.item() / batch_size
            train_losses.append(average_loss)
            loss.backward()
            optimizer.step()
            t.set_postfix({
                'trainloss': '{:.6f}'.format(average_loss),
                'epoch': '{:02d}'.format(epoch)
            })

        net.eval()
        t = tqdm(valid_dataloader, leave=False, total=len(valid_dataloader))

        for batch_idx, sample in enumerate(t):
            inputs, targets = sample['input'], sample['target']
            inputs = inputs.to(device)
            targets = targets.to(device)
            pred = tensorify(net(inputs)[0])
            pred = torch.squeeze(pred, 0)
            loss = loss_f(pred, targets)
            average_loss = loss.item() / batch_size
            val_losses.append(average_loss)
            t.set_postfix({
                'validloss': '{:.6f}'.format(average_loss),
                'epoch': '{:02d}'.format(epoch)
            })
        avg_train_losses.append(np.average(train_losses))
        avg_val_losses.append(np.average(val_losses))

        board.add_scalar('TrainLoss', avg_train_losses[-1], epoch)
        board.add_scalar('ValidLoss', avg_val_losses[-1], epoch)

        # use scheduler ?

        return net, avg_train_losses, avg_val_losses


if __name__ == '__main__':
    net, _, _ = trainer()