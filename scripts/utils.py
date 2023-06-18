from typing import List

from torchaffectnet.const import ID2EXPRESSION


def exclude_id(exclude_labels: List[int]):
    base_id2label = ID2EXPRESSION
    for label in exclude_labels:
        del base_id2label[label]
    id2label = {}
    for i, v in enumerate(base_id2label.values()):
        id2label[i] = v
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id
