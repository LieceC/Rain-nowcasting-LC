from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Models.ConvLSTM import ConvLSTM
from dataset import MeteoDataset
from eval import *
from sampler import indices_except_undefined_sampler, CustomSampler
from utils import *


def trainer(input_shape=(128, 128), input_dim=1, hidden_dim=64, kernel_size=3, input_length=12, output_length=12,
            batch_size=2, epochs=100):
    torch.manual_seed(1)
    board = SummaryWriter("runs/"+str(batch_size)+"_"+str(epochs)+"_lr"+str(10e-6)+"_weight_decay"+str(1e-7))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = ConvLSTM(input_shape=input_shape,
                   input_dim=input_dim,
                   hidden_dim=hidden_dim,
                   kernel_size=kernel_size,
                   device=device)
    net.to(device)
    train = MeteoDataset(rain_dir='../data/rainmap/train', input_length=input_length,
                         output_length=output_length, temporal_stride=input_length, dataset='train')
    train_sampler = CustomSampler(indices_except_undefined_sampler(train), train)

    val = MeteoDataset(rain_dir='../data/rainmap/val', input_length=input_length,
                       output_length=output_length, temporal_stride=input_length, dataset='valid')
    val_sampler = CustomSampler(indices_except_undefined_sampler(val), val)

    train_dataloader = DataLoader(train, batch_size=batch_size, sampler=train_sampler)
    valid_dataloader = DataLoader(val, batch_size=batch_size, sampler=val_sampler)

    optimizer = torch.optim.Adam(net.parameters(), lr=5*10e-6, betas=(0.9, 0.999), weight_decay=1e-7)
    avg_train_losses = []
    avg_val_losses = []

    thresholds_in_mmh = [0.1, 1, 2.5]  # CRF over 1h

    cur_epoch = 0
    for epoch in range(cur_epoch, epochs):
        confusion_matrix = {}
        for thresh in thresholds_in_mmh:
            confusion_matrix[str(thresh)] = {'TP': [0] * output_length,
                                             'FP': [0] * output_length, 'FN': [0] * output_length}
        train_losses = []
        val_losses = []
        net.train()
        t = tqdm(train_dataloader, leave=False, total=len(train_dataloader))
        for batch_idx, sample in enumerate(t):
            inputs, targets = sample['input'], sample['target']
            # [Batch, sequence, Channel, Height, Width]
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            pred = net(inputs)
            mask = compute_weight_mask(targets)
            loss = 0.0005*(weighted_mse_loss(pred, targets, mask) + weighted_mae_loss(pred, targets, mask))
            average_loss = loss.item() / batch_size
            train_losses.append(average_loss)
            loss.backward()
            optimizer.step()

            t.set_postfix({
                'trainloss': '{:.6f}'.format(average_loss),
                'epoch': '{:02d}'.format(epoch)
            })
        save = "checkpoint/model12_at_{}.pth".format(epoch)
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
            pred = net(inputs)
            mask = compute_weight_mask(targets)
            loss = 0.0005 * (weighted_mse_loss(pred, targets, mask) + weighted_mae_loss(pred, targets, mask))
            average_loss = loss.item() / batch_size
            val_losses.append(average_loss)
            t.set_postfix({
                'validloss': '{:.6f}'.format(average_loss),
                'epoch': '{:02d}'.format(epoch)
            })
            for thresh in thresholds_in_mmh:
                conf_mat_batch = compute_confusion(pred, targets, thresh)
                confusion_matrix = add_confusion_matrix_on_batch(confusion_matrix, conf_mat_batch, thresh)

        if epoch > 4:
             scores_evaluation = model_evaluation(confusion_matrix)
             print("[Validation] metrics_scores : ", scores_evaluation)
             avg_train_losses.append(np.average(train_losses))
             avg_val_losses.append(np.average(val_losses))
             board.add_scalar('TrainLoss', avg_train_losses[-1], epoch)
             board.add_scalar('ValidLoss', avg_val_losses[-1], epoch)
             for thresh_key in scores_evaluation:
                 for metric_key in scores_evaluation[thresh_key]:
                     for time_step in scores_evaluation[thresh_key][metric_key]:
                         board.add_scalar(metric_key + "_" + thresh_key + "_time_step_" + time_step,
                                        scores_evaluation[thresh_key][metric_key][time_step],
                                        epoch)

    return net, avg_train_losses, avg_val_losses


if __name__ == '__main__':
    net, _, _ = trainer()

