""" Cross Entropy Loss

"""


# region Imported Dependencies
import warnings
from typing import List, Union, Tuple
import torch
from torch import nn
# endregion Imported Dependencies


# region Functions
def _expand_onehot_labels(a_labels: torch.Tensor, a_label_weights: torch.Tensor, a_label_channels: int,
                          a_ignore_index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bin_labels = a_labels.new_full((a_labels.size(0), a_label_channels), 0)
    valid_mask = (a_labels >= 0) & (a_labels != a_ignore_index)
    inds = torch.nonzero(valid_mask & (a_labels < a_label_channels), as_tuple=False)

    if inds.numel() > 0:
        bin_labels[inds, a_labels[inds]] = 1

    valid_mask = valid_mask.view(-1, 1).expand(a_labels.size(0), a_label_channels).float()

    if a_label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = a_label_weights.view(-1, 1).repeat(1, a_label_channels)
        bin_label_weights *= valid_mask
    return bin_labels, bin_label_weights, valid_mask


def binary_cross_entropy(a_pred: torch.Tensor, a_label: torch.Tensor, a_weight: torch.Tensor = None,
                         a_reduction: str = 'mean', a_avg_factor: int = None, a_class_weight: List[float] = None,
                         a_ignore_index: int = -100, a_avg_non_ignore: bool = False) -> torch.Tensor:
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if a_ignore_index is None else a_ignore_index

    if a_pred.dim() != a_label.dim():
        _expand_onehot_labels(a_labels=a_label, a_label_weights=a_weight, a_label_channels=a_pred.size(-1),
                              a_ignore_index=ignore_index)
    else:
        NotImplemented
# endregion Functions


class CrossEntropyLoss(nn.Module):
    def __init__(self, a_use_sigmoid: bool = False, a_use_mask: bool = False, a_reduction: str = 'mean',
                 a_class_weight: List[float] = None, a_ignore_index: int = None, a_loss_weight: float = 1.0,
                 a_avg_non_ignore: bool = False) -> None:
        super(CrossEntropyLoss, self).__init__()
        assert (a_use_sigmoid is False) or (a_use_mask is False)
        self.use_sigmoid: bool = a_use_sigmoid
        self.use_mask: bool = a_use_mask
        self.reduction: str = a_reduction
        self.class_weight: Union[List[float], None] = a_class_weight
        self.ignore_index: Union[int, None] = a_ignore_index
        self.loss_weight: float = a_loss_weight
        self.avg_non_ignore: bool = a_avg_non_ignore
        if (self.ignore_index is not None) and not self.avg_non_ignore and self.reduction == 'mean':
            warnings.warn('Default `a_avg_non_ignore` is False, if you would like to ignore the certain label and '
                          'average loss over non-ignore labels, which is the same with PyTorch official cross_entropy,'
                          ' set `a_avg_non_ignore=True`.')

        if self.use_sigmoid:
            NotImplemented
        elif self.use_mask:
            NotImplemented
        else:
            NotImplemented

    def forward(self):
        NotImplemented