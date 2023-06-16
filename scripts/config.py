from dataclasses import Field, dataclass, fields, field
from typing import List, Literal, Dict, Any, get_type_hints
from torchaffectnet.const import ID2EXPRESSION
from omegaconf import OmegaConf


@dataclass
class ModelConfig:
    model_ckpt: str = 'google/vit-base-patch16-224-in21k'


class ContrastiveModelConfig(ModelConfig):
    z_dims: int = 128


@dataclass
class TrainConfig:
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    logging_strategy: Literal['steps', 'epoch'] = 'steps'
    logging_steps: int | None = 1000


@dataclass
class DataConfig:
    images_root: str = ''
    train_csv: str = ''
    val_csv: str = ''
    train_invalid_files: List[str] = field(default_factory=list)
    val_invalid_files: List[str] = field(default_factory=list)
    exclude_labels: List[int] = field(default_factory=lambda: [8, 9, 10])


@dataclass
class WandbConfig:
    project: str = ''
    group: str = ''


@dataclass
class _ContrastiveExpConfig:
    name: str = ''
    type: Literal['contrastive'] = 'contrastive'
    supervised: bool = True
    label: Literal['expression', 'categorical-valence',
                   'valence'] = 'expression'
    model: ContrastiveModelConfig = ContrastiveModelConfig()
    train: TrainConfig = TrainConfig()
    data: DataConfig = DataConfig()
    wandb: WandbConfig = WandbConfig()


@dataclass
class _FinetuningExpConfig:
    name: str = ''
    type: Literal['finetuning'] = 'finetuning'
    target: Literal['expression', 'valence',
                    'arousal', 'valence-arousal'] = 'expression'
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    data: DataConfig = DataConfig()
    wandb: WandbConfig = WandbConfig()


@dataclass
class ContrastiveExpConfig:
    exp: _ContrastiveExpConfig = _ContrastiveExpConfig()


@dataclass
class FinetuningExpConfig:
    exp: _FinetuningExpConfig = _FinetuningExpConfig()


cfg_label = ('expression', 'categorical-valence', 'valence')
cfg_target = ('expression', 'valence', 'arousal', 'valence-arousal')


def validate_cfg(cfg: ContrastiveExpConfig | FinetuningExpConfig):
    if cfg.exp.type == 'contrastive':
        if not cfg.exp.label in cfg_label:
            print(f'config.label must be {cfg_label}.')
            exit(-1)

    if cfg.exp.type == 'finetuning':
        if not cfg.exp.target in cfg_target:
            print(f'config.target must be {cfg_target}.')
            exit(-1)
