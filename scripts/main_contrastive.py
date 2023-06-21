import os
from typing import Any, Dict, Tuple
import wandb
from evaluate import evaluate
from trainer import SupConTrainer
from dataset import AffectNetDatasetForSupConWithValence
from model import load_model
from options import options, Options
from config import ContrastiveExpConfig, validate_cfg
from omegaconf import OmegaConf
import hydra
from torchaffectnet.collators import ContrastiveCollator
from torchaffectnet import AffectNetDatasetForSupCon, AffectNetDataset
from torchvision.transforms import (Compose,
                                    Normalize,
                                    Resize,
                                    RandomResizedCrop,
                                    RandomHorizontalFlip,
                                    RandomApply,
                                    ColorJitter,
                                    RandomGrayscale,
                                    ToTensor)
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments
from utils import try_finish_wandb


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

    if cfg.exp.label == 'categorical-valence':
        dataset = AffectNetDatasetForSupConWithValence(cfg.exp.data.train_csv,
                                                       cfg.exp.data.images_root,
                                                       transform=transform,
                                                       exclude_label=cfg.exp.data.exclude_labels,
                                                       invalid_files=cfg.exp.data.exclude_labels)
    elif cfg.exp.label == 'expression':
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
    if cfg.exp.type != 'contrastive':
        print('type must be "contrastive".')
        exit(-1)
    validate_cfg(cfg)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, cfg.exp.cuda_devices))
    
    import torch
    from torch.nn.parallel import DataParallel
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.device_count())

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), f'../outputs/{cfg.exp.name}'))
    os.makedirs(output_dir, mode=0o777)
    print(output_dir)
    
    cfg_yaml = OmegaConf.to_yaml(cfg)
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        f.write(cfg_yaml)

    print(cfg_yaml)
    opt = options(cfg)
    print(opt)
    feature_extractor, model = load_model(cfg, opt)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    train_dataset = prepare_dataset(cfg, opt, feature_extractor)

    # Train
    print('Training...')
    wandb.init(project=cfg.exp.wandb.project,
               group=cfg.exp.wandb.group, name=cfg.exp.name)

    trainer_args = TrainingArguments(
        os.path.join(output_dir, 'checkpoints'),
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
    
    if isinstance(trainer.model, DataParallel):
        trainer.model = trainer.model.module
    trainer.save_model(os.path.join(output_dir, 'model'))

    # Evaluate
    print('Evaluating...')
    evaluate(cfg.exp.data.images_root,
             cfg.exp.data.val_csv,
             cfg.exp.data.exclude_labels,
             cfg.exp.data.val_invalid_files,
             model.module if isinstance(model, DataParallel) else model,
             feature_extractor,
             20,
             device,
             output_dir,
             False,
             wandb_log=True,
             wandb_resume=False,
             wandb_proj=cfg.exp.wandb.project,
             wandb_group=cfg.exp.wandb.group,
             wandb_name=cfg.exp.name,
             after_train=True)
    
    # wandb.finish()
    try_finish_wandb()


if __name__ == '__main__':
    main()
