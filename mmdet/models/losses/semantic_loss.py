import torch.nn as nn
import torch.nn.functional as F
import torch

from ..builder import LOSSES
from .utils import weight_reduce_loss


def cross_entropy(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None):
    """Calculate the CrossEntropy loss.
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def semantic_exactly_one(pred):
    """Semantic loss.
    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).

    Returns:
        torch.Tensor: The calculated semantic loss
    """
    prob = F.sigmoid(pred)
    wmc_tmp = torch.zeros_like(prob)
    # exactly one semantic loss based on definition 1
    for i in range(pred.shape[1]):
        one_situation = torch.ones_like(pred).scatter_(1, torch.zeros_like(pred[:, 0]).fill_(i).unsqueeze(-1).long(), 0)
        wmc_tmp[:, i] = torch.abs((one_situation - prob).prod(dim=1))
    _log_wmc_tmp = -1.0 * torch.log(wmc_tmp.sum(dim=1))
    return _log_wmc_tmp


@LOSSES.register_module()
class CESemanticLoss(nn.Module):
    """Cross entropy plus Semantic loss.
    Args:
        cls_score (torch.Tensor): The prediction with shape (N, \*).
        label (torch.Tensor): The gt label with shape (N, \*).

    Returns:
        torch.Tensor: The calculated CE with semantic loss for labelled und unlablled samples
    """
def __init__(self, ):
    super(CESemanticLoss, self).__init__()

def forward(self,
            cls_score,
            label):
    labelled_examples = label.sum(dim=1)
    unlabelled_examples = 1.0 - labelled_examples
    CE = torch.multiply(labelled_examples, self.loss_weight * self.cross_entropy(cls_score, label))
    semantic = 0.0005 * torch.multiply(labelled_examples, semantic_exactly_one(cls_score)) + \
               0.0005 * torch.multiply(unlabelled_examples, semantic_exactly_one(cls_score))
    CE_Semantic_Loss = torch.mean(torch.sum(torch.add(CE, semantic)))
    return CE_Semantic_Loss
