import torch
import torch.nn as nn
import torch.nn.functional as F

criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    criterion.cuda()
else:
    try:
        criterion.to('mps')
    except Exception as e:
        pass

def CNNLoss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    loss = criterion(y_pred, y_true)
    return loss


def RegLoss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    no_object_loss = torch.pow((1 - y_true[:, 0]) * y_pred[:, 0], 2).mean()
    object_loss = torch.pow((y_true[:, 0]) * (y_pred[:, 0] - 1), 2).mean()

    reg_loss = (
        y_true[:, 0] * (torch.pow(y_true[:, 1:5] - y_pred[:, 1:5], 2).sum(1))).mean()

    loss = no_object_loss + object_loss + reg_loss
    return loss


def HingeLoss(y_pred, y_true):
    # y_pred: [batch_size, num_class]
    # y_true: [batch_size, ]
    num_true = y_true.shape[0]
    corrects = y_pred[range(num_true), y_true].unsqueeze(0).T

    margin = 1.0
    margins = y_pred - corrects + margin
    loss = torch.sum(torch.max(margins, 1)[0]) / len(y_true)

    # regressions
    # reg = 1e-3
    # loss += reg * torch.sum(weight ** 2)
    return loss
