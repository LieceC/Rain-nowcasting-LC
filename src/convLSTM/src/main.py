import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Models.ConvLSTM import ConvLSTM, torch
from dataset import MeteoDataset
from utils import tensorify


def trainer(input_shape=(128, 128), input_dim=1, hidden_dim=64, kernel_size=3, input_length=2, output_length=2,
            batch_size=2, epochs=20):
    torch.manual_seed(1)
    board = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ConvLSTM(input_shape=input_shape,
                   input_dim=input_dim,
                   hidden_dim=hidden_dim,
                   kernel_size=kernel_size,
                   device=device)
    net.to(device)

    train = MeteoDataset(rain_dir='../data/rainmap/train', input_length=input_length,
                         output_length=output_length, temporal_stride=input_length, dataset='train')
    # train_sampler = CustomSampler(indices_except_undefined_sampler(train), train)

    val = MeteoDataset(rain_dir='../data/rainmap/val', input_length=input_length,
                       output_length=output_length, temporal_stride=input_length, dataset='valid')
    # val_sampler = CustomSampler(indices_except_undefined_sampler(val), val)

    train_dataloader = DataLoader(train, batch_size=batch_size)
    valid_dataloader = DataLoader(val, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=10e-8, betas=(0.9, 0.999))
    loss_f = torch.nn.BCEWithLogitsLoss()

    avg_train_losses = []
    avg_val_losses = []

    thresholds_in_mmh = [0, 0.1, 1, 2.5]  # CRF over 1h
    thresholds_in_cent_mm = [100 * k / 12 for k in thresholds_in_mmh]

    # thresholds_normalized = np.log(np.array(thresholds_in_cent_mm) + 1) / train.norm_factor

    cur_epoch = 0
    for epoch in range(cur_epoch, epochs):
        print(epoch)
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
            # print(pred.size())
            # print(targets.size())
            loss = loss_f(pred, targets)
            average_loss = loss.item() / batch_size
            train_losses.append(average_loss)
            loss.backward()
            optimizer.step()

            t.set_postfix({
                'trainloss': '{:.6f}'.format(average_loss),
                'epoch': '{:02d}'.format(epoch)
            })
        save = "checkpoint.pth"
        state = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, save)
        net.eval()
        t = tqdm(valid_dataloader, leave=False, total=len(valid_dataloader))

        for batch_idx, sample in enumerate(t):
            inputs, targets = sample['input'], sample['target']
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
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

    return net, avg_train_losses, avg_val_losses


if __name__ == '__main__':
    net, _, _ = trainer()
