import torch
from torchaffectnet.datasets import AffectNetDatasetForSupCon


class AffectNetDatasetForSupConWithValence(AffectNetDatasetForSupCon):
    def labeling(self, idx):
        target = torch.tensor(self.df['valence'][idx])
        if target < -0.5:
            return torch.tensor(0)
        elif target > 0.5:
            return torch.tensor(2)
        else:
            return torch.tensor(1)
