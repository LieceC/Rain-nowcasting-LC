from torch.utils.data import DataLoader
from tqdm import tqdm

from Models.ConvLSTM import ConvLSTM
from dataset import MeteoDataset
from eval import *
from src.sampler import CustomSampler, indices_except_undefined_sampler
from utils import *


def eval(input_shape=(128, 128), input_dim=1, hidden_dim=64, kernel_size=3, input_length=2, output_length=2,
         batch_size=2):
    torch.manual_seed(1)
    # checkpoint = torch.load("checkpoints/model_at_epoch1.pth")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ConvLSTM(input_shape=input_shape,
                   input_dim=input_dim,
                   hidden_dim=hidden_dim,
                   kernel_size=kernel_size,
                   device=device)
    net.to(device)
    # net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    val = MeteoDataset(rain_dir='../data/rainmap/val', input_length=input_length,
                       output_length=output_length, temporal_stride=input_length, dataset='valid')
    val_sampler = CustomSampler(indices_except_undefined_sampler(val), val)
    valid_dataloader = DataLoader(val, batch_size=batch_size, sampler=val_sampler)

    thresholds_in_mmh = [0.1, 1, 2.5]  # CRF over 1h

    confusion_matrix = {}
    for thresh in thresholds_in_mmh:
        confusion_matrix[str(thresh)] = {'TP': [0] * output_length,
                                         'FP': [0] * output_length, 'FN': [0] * output_length}
    val_losses = []
    t = tqdm(valid_dataloader, leave=False, total=len(valid_dataloader))

    for batch_idx, sample in enumerate(t):
        inputs, targets = sample['input'], sample['target']
        inputs = inputs.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)
        pred = net(inputs)[0]
        mask = compute_weight_mask(targets)
        loss = weighted_mse_loss(pred, targets, mask) + weighted_mae_loss(pred, targets, mask)
        average_loss = loss.item() / batch_size
        val_losses.append(average_loss)
        t.set_postfix({
            'validloss': '{:.6f}'.format(average_loss),
        })
        for thresh in thresholds_in_mmh:
            conf_mat_batch = compute_confusion(pred, targets, thresh)
            confusion_matrix = add_confusion_matrix_on_batch(confusion_matrix, conf_mat_batch, thresh)

    scores_evaluation = model_evaluation(confusion_matrix)
    print("[Validation] metrics_scores : ", scores_evaluation)
    print(np.average(val_losses))


if __name__ == '__main__':
    eval()
