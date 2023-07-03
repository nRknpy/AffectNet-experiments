import torch
from torchaffectnet.datasets import AffectNetDatasetForSupCon
from typing import List, Tuple


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
        super().__init__(csvfile, root, transform1, transform2, exclude_label, return_labels, crop, invalid_files)
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