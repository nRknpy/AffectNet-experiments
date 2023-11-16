from typing import Dict, Any

import torch
from torch.utils.data import Dataset
from torchaffectnet.datasets import AffectNetDatasetForSupCon
from torchaffectnet.collators import Collator
from typing import List, Tuple


categorical_valence_id2label = {
    0: 'valence < -0.5',
    1: '-0.5 <= valence <= 0.5',
    2: '0.5 < valence',
}


class AffeLangDataset(AffectNetDatasetForSupCon):
    def labeling(self, idx):
        return torch.tensor([self.df['valence'][idx], self.df['arousal'][idx], self.df['expression'][idx]], dtype=torch.float)


class AlternatingDataset(Dataset):
    def __init__(self, dataset1, dataset2, batch_size, alter_steps):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.batch_size = batch_size
        self.alter_steps = alter_steps

    def __len__(self):
        return max(len(self.dataset1), len(self.dataset2))

    def __getitem__(self, idx):
        if (idx // (self.batch_size * self.alter_steps)) % 2 == 0:
            data = self.dataset1[idx]
            dataset_id = 0
        else:
            data = self.dataset2[idx]
            dataset_id = 1
        return data, dataset_id


class AlternatingContrastiveCollator(Collator):
    def __init__(self, return_labels=[True, True]) -> None:
        super().__init__()
        self.return_labels = return_labels

    def collate_fn(self, examples) -> Dict[str, Any]:
        data, dataset_id = zip(*examples)
        dataset_id = dataset_id[0]
        if self.return_labels[dataset_id]:
            imgs, targets = zip(*data)
            targets = torch.stack(targets)
        else:
            imgs = data

        imgs1, imgs2 = zip(*imgs)
        imgs1 = torch.stack(imgs1)
        imgs2 = torch.stack(imgs2)

        pixel_values = torch.cat([imgs1, imgs2])

        if self.return_labels[dataset_id]:
            return {'pixel_values': pixel_values, 'labels': targets, 'dataset_id': dataset_id}
        else:
            return {'pixel_values': pixel_values, 'dataset_id': dataset_id}


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


class AffectNetDatasetForSupConWithLandmark(AffectNetDatasetForSupCon):
    def __init__(self, csvfile: str, root: str, transform1, transform2, exclude_label: Tuple[int] = ..., return_labels: bool = True, crop: bool = False, invalid_files: List[str] = None):
        super().__init__(csvfile, root, transform1, transform2,
                         exclude_label, return_labels, crop, invalid_files)
        landmarks = self.df['facial_landmarks']
        face_x = self.df['face_x']
        face_y = self.df['face_y']
        face_width = self.df['face_width']
        face_height = self.df['face_height']
        landmarks_split = landmarks.str.split(';')
        x_coordinates = []
        y_coordinates = []
        for landmark, x_min, y_min, w, h in zip(landmarks_split, face_x, face_y, face_width, face_height):
            coordinates = [float(val) for val in landmark]
            x = torch.tensor(coordinates[::2], dtype=torch.float)
            y = torch.tensor(coordinates[1::2], dtype=torch.float)
            x = (x - x_min) / w
            y = (y - y_min) / h
            x_coordinates.append(x)
            y_coordinates.append(y)
        self.x_landmarks = torch.stack(x_coordinates)
        self.y_landmarks = torch.stack(y_coordinates)

    def labeling(self, idx):
        x_landmark = self.x_landmarks[idx]
        y_landmark = self.y_landmarks[idx]
        return torch.stack([x_landmark, y_landmark])
