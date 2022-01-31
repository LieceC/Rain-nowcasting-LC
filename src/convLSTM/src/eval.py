import torch


def add_confusion_matrix_on_batch(confusion_matrix, confusion_matrix_on_batch, threshold):
    for key in confusion_matrix_on_batch:
        for time_step in range(len(confusion_matrix_on_batch[key])):
            confusion_matrix[str(threshold)][key][time_step] += confusion_matrix_on_batch[key][time_step]
    return confusion_matrix


def compute_confusion(y_pred, y, threshold):
    diff = 2 * torch.where(y_pred >= threshold, 1.0, 0.0) - torch.where(y >= threshold, 1.0, 0.0)
    return {'TP': torch.sum(diff == 1, dim=(0, 2, 3, 4)).tolist(),
            'FP': torch.sum(diff == 2, dim=(0, 2, 3, 4)).tolist(),
            'FN': torch.sum(diff == -1, dim=(0, 2, 3, 4)).tolist()}


def model_evaluation(confusion_matrix):
    scores = {}
    for thresh_key in confusion_matrix:
        scores[thresh_key] = {}
        scores[thresh_key]['f1'] = {}
        scores[thresh_key]['ts'] = {}
        scores[thresh_key]['bias'] = {}
        for time_step in range(len(confusion_matrix[thresh_key]['TP'])):
            scores[thresh_key]['f1']["t+" + str(time_step + 1)] = compute_f1_score(confusion_matrix[thresh_key],
                                                                                   time_step)
            scores[thresh_key]['ts']["t+" + str(time_step + 1)] = compute_ts_score(confusion_matrix[thresh_key],
                                                                                   time_step)
            scores[thresh_key]['bias']["t+" + str(time_step + 1)] = compute_bias_score(
                confusion_matrix[thresh_key], time_step)
    return scores


def compute_f1_score(conf_mat, time_step):
    precision = conf_mat['TP'][time_step] / (
            conf_mat['TP'][time_step] + conf_mat['FP'][time_step])
    recall = conf_mat['TP'][time_step] / (
            conf_mat['TP'][time_step] + conf_mat['FN'][time_step])
    metric_score = 2 * precision * recall / (precision + recall)

    return round(metric_score, 3)


def compute_ts_score(conf_mat, time_step):
    metric_score = conf_mat['TP'][time_step] / (
            conf_mat['TP'][time_step] + conf_mat['FP'][time_step] + conf_mat['FN'][
        time_step])

    return round(metric_score, 3)


def compute_bias_score(conf_mat, time_step):
    metric_score = (conf_mat['TP'][time_step] + conf_mat['FP'][time_step]) / (
            conf_mat['TP'][time_step] + conf_mat['FN'][time_step])

    return round(metric_score, 3)
