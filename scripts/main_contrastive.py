from typing import Any, Dict, Tuple
import torch
import wandb
import os
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments
from torchvision.transforms import (Compose,
                                    Normalize,
                                    Resize,
                                    RandomResizedCrop,
                                    RandomHorizontalFlip,
                                    RandomApply,
                                    ColorJitter,
                                    RandomGrayscale,
                                    ToTensor)
from torchaffectnet import AffectNetDatasetForSupCon, AffectNetDataset
from torchaffectnet.collators import ContrastiveCollator
import hydra
from omegaconf import OmegaConf

from config import ContrastiveExpConfig, validate_cfg
from options import options, Options
from model import load_model
from dataset import AffectNetDatasetForSupConWithValence
from trainer import SupConTrainer
from evaluate import evaluate


def prepare_dataset(cfg: ContrastiveExpConfig, opt: Options, feature_extractor: Tuple[ViTFeatureExtractor, Dict[str, Any]] | ViTFeatureExtractor):
    normalize = Normalize(mean=feature_extractor.image_mean,
                          std=feature_extractor.image_std)

    transform = Compose([
        RandomResizedCrop(size=tuple(
            feature_extractor.size.values()), scale=(0.2, 1.)),
        RandomHorizontalFlip(),
        RandomApply([
            ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        RandomGrayscale(p=0.2),
        ToTensor(),
        normalize
    ])

    if cfg.exp.label == 'expression':
        dataset = AffectNetDatasetForSupConWithValence(cfg.exp.data.train_csv,
                                                       cfg.exp.data.images_root,
                                                       transform=transform,
                                                       exclude_label=cfg.exp.data.exclude_labels,
                                                       invalid_files=cfg.exp.data.exclude_labels)
    elif cfg.exp.label == 'categorical-valence':
        dataset = AffectNetDatasetForSupCon(cfg.exp.data.train_csv,
                                            cfg.exp.data.images_root,
                                            transform=transform,
                                            exclude_label=cfg.exp.data.exclude_labels,
                                            invalid_files=cfg.exp.data.exclude_labels)
    elif cfg.exp.label == 'valence':
        print('CL with continuous valence is not implemented.')
        exit(-1)
    return dataset


@hydra.main(version_base=None, config_path='../', config_name='config')
def main(cfg: ContrastiveExpConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']

    print(OmegaConf.to_yaml(cfg))
    validate_cfg(cfg)
    opt = options(cfg)
    print(opt)
    feature_extractor, model = load_model(cfg, opt)
    train_dataset = prepare_dataset(cfg, opt, feature_extractor)

    # Train
    wandb.init(project=cfg.exp.wandb.project,
               group=cfg.exp.wandb.group, name=cfg.exp.name)

    trainer_args = TrainingArguments(
        os.path.join(output_dir, cfg.exp.name),
        save_strategy='epoch',
        learning_rate=cfg.exp.train.learning_rate,
        per_device_train_batch_size=int(
            cfg.exp.train.batch_size / torch.cuda.device_count()),
        num_train_epochs=cfg.exp.train.num_epochs,
        weight_decay=cfg.exp.train.weight_decay,
        logging_strategy=cfg.exp.train.logging_strategy,
        logging_steps=cfg.exp.train.logging_steps,
        remove_unused_columns=False,
        report_to='wandb',
    )

    trainer = SupConTrainer(
        model,
        trainer_args,
        train_dataset=train_dataset,
        data_collator=ContrastiveCollator(return_labels=opt.return_labels),
        tokenizer=feature_extractor,
    )

    trainer.train()

    # Evaluate
    evaluate(cfg.exp.data.images_root, cfg.exp.data.val_csv, cfg.exp.data.exclude_labels,
             cfg.exp.data.val_invalid_files,
             model, feature_extractor, 20, device, output_dir, False)


if __name__ == '__main__':
    main()
