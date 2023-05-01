import numpy as np
import torch
from functools import partial
from sklearn.metrics import *


def get_metric_func(final_metric, task_type):
    try:
        metric = eval(final_metric)
    except Exception as e:
        print(e);
        if 'accuracy' in final_metric:
            metric = zero_one_error
        elif 'zero_one' in final_metric:
            metric = zero_one_error
        elif 'log' in final_metric:
            metric = log_loss
        elif 'binary' in final_metric:
            metric = log_loss
        elif 'mae' in final_metric:
            metric = mean_absolute_error
        elif 'l1' in final_metric:
            metric = mean_absolute_error
        elif 'l2' in final_metric:
            metric = mean_squared_error
        elif 'f1' in final_metric:
            metric = inv_f1_score            
        elif 'auc' in final_metric:
            metric = inv_roc_auc_score
        elif 'map' in final_metric:
            metric = inv_map_score
        elif 'continuous' in task_type:
            metric = mean_squared_error
        elif 'label' in task_type:
            metric = log_loss
        else:
            metric = zero_one_error;

    return metric


def OHE(raw_labels, n):
    ohe_labels = np.eye(n)[raw_labels]
    return ohe_labels

class RelativeError(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(RelativeError, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def __call__(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms


def loss_metric(solution, prediction, loss_class):
    loss = loss_class()
    return loss(torch.Tensor(solution), torch.Tensor(prediction)).data.item()


def l2_relative_error(solution, prediction, size_average=True):
    return loss_metric(
        solution, prediction, partial(RelativeError, size_average=size_average)
    )

def zero_one_error(solution, prediction):
    y = np.argmax(solution, axis=1)
    y_hat = np.argmax(prediction, axis=1)
    return zero_one_loss(y, y_hat)


def inv_f1_score(solution, prediction):
    y = np.argmax(solution, axis=1)
    y_hat = np.argmax(prediction, axis=1)
    return 1 - f1_score(y, y_hat, average = 'macro')


def inv_auroc_score(solution, prediction):
    y = solution > 0.5
    y_hat = prediction
    try:
        return 1.0 - roc_auc_score(y, y_hat, average = 'macro')
    except:
        return 1.0 - roc_auc_score(y, y_hat, average = 'samples')
    

def inv_map_score(solution, prediction):
    y = solution
    y_hat = prediction
    return 1.0 - average_precision_score(y, y_hat, average = 'micro')


def nll_score(solution, prediction):
    return loss_metric(prediction, solution, partial(torch.nn.BCELoss))


def false_negative_rate(solution, prediction):
    solution = solution.reshape(solution.shape[0], -1)
    prediction = prediction.reshape(prediction.shape[0], -1)
    y = solution > 0.5
    y_hat = prediction > 0.5
    if len(y_hat.shape) == 2:
        y_hat = y_hat.reshape(1, *y_hat.shape)
    if len(y.shape) == 2:
        y = y.reshape(1, *y.shape)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(y.shape[0]):
        TP += (y_hat[i][y[i] == 1] == 1).sum()
        TN += (y_hat[i][y[i] == 0] == 0).sum()
        FP += (y_hat[i][y[i] == 0] == 1).sum()
        FN += (y_hat[i][y[i] == 1] == 0).sum()
    return FN / (FN + TP)

