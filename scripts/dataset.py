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


class AlternatingDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return max(len(self.dataset1), len(self.dataset2))

    def __getitem__(self, idx):
        data1 = self.dataset1[idx]
        data2 = self.dataset2[idx]
        return data1, data2


class AlternatingCollator(Collator):
    def __init__(self, return_labels=[True, True]) -> None:
        super().__init__()
        self.return_labels1 = return_labels[0]
        self.return_labels2 = return_labels[1]
    
    def collate_fn(self, examples) -> Dict[str, Any]:
        data1, data2 = zip(*examples)
        if self.return_labels1:
            data1_imgs, data1_targets = zip(*data1)
            data1_targets = torch.stack(data1_targets)
        else:
            data1_imgs = data1
        
        if self.return_labels2:
            data2_imgs, data2_targets = zip(*data2)
            data2_targets = torch.stack(data2_targets)
        else:
            data2_imgs = data2
        
        data1_imgs1, data1_imgs2 = zip(*data1_imgs)
        data1_imgs1 = torch.stack(data1_imgs1)
        data1_imgs2 = torch.stack(data1_imgs2)
        
        data2_imgs1, data2_imgs2 = zip(*data2_imgs)
        data2_imgs1 = torch.stack(data2_imgs1)
        data2_imgs2 = torch.stack(data2_imgs2)
        
        pixel_values1 = torch.cat([data1_imgs1, data1_imgs2])
        pixel_values2 = torch.cat([data2_imgs1, data2_imgs2])
        
        output = []
        if self.return_labels1:
            output.append(
                {'pixel_values': pixel_values1, 'labels': data1_targets}
            )
        else:
            output.append({'pixel_values': pixel_values1})
        if self.return_labels2:
            output.append(
                {'pixel_values': pixel_values2, 'labels': data2_targets}
            )
        else:
            output.append({'pixel_values': pixel_values2})
        
        return output


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
            x = (x - x_min)/w
            y = (y - y_min)/h
            x_coordinates.append(x)
            y_coordinates.append(y)
        self.x_landmarks = torch.stack(x_coordinates)
        self.y_landmarks = torch.stack(y_coordinates)

    def labeling(self, idx):
        x_landmark = self.x_landmarks[idx]
        y_landmark = self.y_landmarks[idx]
        return torch.stack([x_landmark, y_landmark])
