from torch.utils.data import DataLoader

from Models.ConvLSTM import ConvLSTM
from dataset import MeteoDataset
from eval import *
from sampler import CustomSampler, indices_except_undefined_sampler
from utils import *


def eval(input_shape=(128, 128), input_dim=1, hidden_dim=64, kernel_size=3, input_length=12, output_length=12,
         batch_size=2):
    torch.manual_seed(1)
    #change to the path of your checkpoint
    checkpoint = torch.load("checkpoint/test_fix2_model12_at_40.pth", map_location=torch.device('cuda'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ConvLSTM(input_shape=input_shape,
                   input_dim=input_dim,
                   output_dim=1,
                   hidden_dim=hidden_dim,
                   kernel_size=kernel_size,
                   device=device)
    net.to(device)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    test = MeteoDataset(rain_dir='../PRAT21/data/rainmap/val',
                        wind_dir=None,
                        input_length=input_length,
                        output_length=output_length,
                        temporal_stride=input_length,
                        dataset='test',
                        recurrent_nn=True)
    test_sampler = CustomSampler(indices_except_undefined_sampler(test), test)
    test_dataloader = DataLoader(test, batch_size=batch_size, sampler=test_sampler)

    thresholds_in_mmh = [0.1, 1, 2.5]  # CRF over 1h
    index_plot = 0
    confusion_matrix = {}
    for thresh in thresholds_in_mmh:
        confusion_matrix[str(thresh)] = {'TP': [0] * output_length,
                                         'FP': [0] * output_length, 'FN': [0] * output_length}
    t = tqdm(test_dataloader, leave=False, total=len(test_dataloader))

    for batch_idx, sample in enumerate(t):
        inputs, targets = sample['input'], sample['target']
        inputs = inputs.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)
        pred = net(inputs)

        for thresh in thresholds_in_mmh:
            conf_mat_batch = compute_confusion(pred, targets, thresh)
            confusion_matrix = add_confusion_matrix_on_batch(confusion_matrix, conf_mat_batch, thresh)
        for k in range(inputs.shape[0]):
            save_gif_2(pred[k], 'images/{}_pred.gif'.format(index_plot))
            save_gif_2(targets[k], 'images/{}_target.gif'.format(index_plot))
            plot_output_gt_colored(pred[k], targets[k], inputs[k], index_plot, 'images/plot_pred_targets')
            index_plot += 1
    scores_evaluation = model_evaluation(confusion_matrix)
    print("[Validation] metrics_scores : ", scores_evaluation)


if __name__ == '__main__':
    eval()
