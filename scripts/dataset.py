import torch
from torchaffectnet.datasets import AffectNetDatasetForSupCon


class AffectNetDatasetForSupConWithCategoricalValence(AffectNetDatasetForSupCon):
    def labeling(self, idx):
        target = torch.tensor(self.df['valence'][idx])
        if target < -0.5:
            return torch.tensor(0)
        elif target > 0.5:
            return torch.tensor(2)
        else:
            return torch.tensor(1)


class AffectNetDatasetForSupConWithValence(AffectNetDatasetForSupCon):
    def labeling(self, idx):
        return torch.tensor(self.df['valence'][idx], dtype=torch.float)


class AffectNetDatasetForSupConWithArousal(AffectNetDatasetForSupCon):
    def labeling(self, idx):
        return torch.tensor(self.df['arousal'][idx], dtype=torch.float)


class AffectNetDatasetForSupConWithValenceArousal(AffectNetDatasetForSupCon):
    def labeling(self, idx):
        return torch.tensor([self.df['valence'][idx], self.df['arousal'][idx]], dtype=torch.float)
