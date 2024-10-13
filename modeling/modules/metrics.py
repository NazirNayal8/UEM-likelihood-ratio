import torch
from torchmetrics.classification import BinaryROC

class FPR95(BinaryROC):
    """
    Calculate false positive rate at the threshold where false positive rate is 0.95
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute(self):
        fpr, tpr, thresholds = super().compute()
        fpr95 = 0        
        idx = torch.argmax((tpr >= 0.95).float())
        fpr95 = fpr[idx]
        return fpr95