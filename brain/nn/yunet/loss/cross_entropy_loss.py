""" Cross Entropy Loss

"""


# region Imported Dependencies
import warnings
from typing import List, Union
from torch import nn
# endregion Imported Dependencies


class CrossEntropyLoss(nn.Module):
    def __init__(self, a_use_sigmoid: bool = False, a_use_mask: bool = False, a_reduction: str = 'mean',
                 a_class_weight: Union[List[float], None] = None, a_ignore_index: Union[int, None] = None,
                 a_loss_weight: float = 1.0, a_avg_non_ignore: bool = False):
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