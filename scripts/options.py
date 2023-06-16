from typing import List, Literal, Dict, Any
from dataclasses import dataclass, field
from torchaffectnet.const import ID2EXPRESSION

from config import ContrastiveExpConfig, FinetuningExpConfig


@dataclass
class Options:
    num_labels: int = 11
    problem_type: Literal['single_label_classification',
                          'regression'] = 'single_label_classification'
    id2label: Dict[int, str] | None = field(default_factory=dict)
    label2id: Dict[str, int] | None = field(default_factory=dict)

    return_labels: bool = True


def contrastive_options(cfg: ContrastiveExpConfig):
    num_labels = cfg.exp.model.z_dims
    problem_type = 'regression'
    id2label = label2id = None
    return_labels = cfg.exp.supervised
    return Options(
        num_labels=num_labels,
        problem_type=problem_type,
        id2label=id2label,
        label2id=label2id,
        return_labels=return_labels
    )


def finetuning_options(cfg: FinetuningExpConfig):
    if cfg.exp.target == 'expression':
        num_labels = 11 - len(cfg.exp.data.exclude_labels)
        problem_type = 'single_label_classification'
        id2label = ID2EXPRESSION
        for label in cfg.exp.data.exclude_labels:
            del id2label[label]
        label2id = {v: k for k, v in id2label.items()}

    elif cfg.exp.target == 'valence' or cfg.exp.target == 'arousal':
        num_labels = 1
        problem_type = 'regression'
        id2label = label2id = None

    elif cfg.exp.target == 'valence-arousal':
        num_labels = 2
        problem_type = 'regression'
        id2label = label2id = None

    return Options(
        num_labels=num_labels,
        problem_type=problem_type,
        label2id=label2id,
        id2label=id2label
    )


def options(cfg: ContrastiveExpConfig | FinetuningExpConfig):
    if cfg.exp.type == 'contrastive':
        return contrastive_options(cfg)
    elif cfg.exp.type == 'finetuning':
        return finetuning_options(cfg)
