import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


from typing import Any, Dict, Tuple
import torch
import wandb
from evaluate import evaluate, compute_rmse, compute_accuracy
from trainer import WeightedLossTrainer, KDEwMSETrainer
from dataset import AffectNetDatasetForSupConWithValence
from model import load_model
from options import options, Options
from config import FinetuningExpConfig, validate_cfg
from omegaconf import OmegaConf
import hydra
from torchaffectnet.collators import Collator
from torchaffectnet import AffectNetDataset
from torchvision.transforms import Compose, ToTensor, Resize, Normalize,RandomAffine
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, EarlyStoppingCallback
from utils import try_finish_wandb


def prepare_dataset(cfg: FinetuningExpConfig, opt: Options, feature_extractor: Tuple[ViTFeatureExtractor, Dict[str, Any]] | ViTFeatureExtractor):
    normalize = Normalize(mean=feature_extractor.image_mean,
                          std=feature_extractor.image_std)

    train_transform = Compose([
        RandomAffine(30),
        Resize(tuple(feature_extractor.size.values())),
        ToTensor(),
        normalize,
    ])
    val_transform = Compose([
        Resize(tuple(feature_extractor.size.values())),
        ToTensor()
    ])

    if cfg.exp.target == 'expression':
        train_dataset = AffectNetDataset(cfg.exp.data.train_csv,
                                         cfg.exp.data.images_root,
                                         transform=train_transform,
                                         exclude_label=cfg.exp.data.exclude_labels,
                                         invalid_files=cfg.exp.data.train_invalid_files,
                                         mode='classification')
        val_dataset = AffectNetDataset(cfg.exp.data.val_csv,
                                       cfg.exp.data.images_root,
                                       transform=val_transform,
                                       exclude_label=cfg.exp.data.exclude_labels,
                                       invalid_files=cfg.exp.data.val_invalid_files,
                                       mode='classification')
    
    elif cfg.exp.target == 'valence':
        train_dataset = AffectNetDataset(cfg.exp.data.train_csv,
                                         cfg.exp.data.images_root,
                                         transform=train_transform,
                                         exclude_label=cfg.exp.data.exclude_labels,
                                         invalid_files=cfg.exp.data.train_invalid_files,
                                         mode='valence')
        val_dataset = AffectNetDataset(cfg.exp.data.val_csv,
                                       cfg.exp.data.images_root,
                                       transform=val_transform,
                                       exclude_label=cfg.exp.data.exclude_labels,
                                       invalid_files=cfg.exp.data.val_invalid_files,
                                       mode='valence')
    
    elif cfg.exp.target == 'arousal':
        train_dataset = AffectNetDataset(cfg.exp.data.train_csv,
                                         cfg.exp.data.images_root,
                                         transform=train_transform,
                                         exclude_label=cfg.exp.data.exclude_labels,
                                         invalid_files=cfg.exp.data.train_invalid_files,
                                         mode='arousal')
        val_dataset = AffectNetDataset(cfg.exp.data.val_csv,
                                       cfg.exp.data.images_root,
                                       transform=val_transform,
                                       exclude_label=cfg.exp.data.exclude_labels,
                                       invalid_files=cfg.exp.data.val_invalid_files,
                                       mode='arousal')
    
    elif cfg.exp.target == 'valence-arousal':
        train_dataset = AffectNetDataset(cfg.exp.data.train_csv,
                                         cfg.exp.data.images_root,
                                         transform=train_transform,
                                         exclude_label=cfg.exp.data.exclude_labels,
                                         invalid_files=cfg.exp.data.train_invalid_files,
                                         mode='valence-arousal')
        val_dataset = AffectNetDataset(cfg.exp.data.val_csv,
                                       cfg.exp.data.images_root,
                                       transform=val_transform,
                                       exclude_label=cfg.exp.data.exclude_labels,
                                       invalid_files=cfg.exp.data.val_invalid_files,
                                       mode='valence-arousal')

    return train_dataset, val_dataset


@hydra.main(version_base=None, config_path='../', config_name='config')
def main(cfg: FinetuningExpConfig):
    if cfg.exp.type != 'finetuning':
        print('type must be "finetuning".')
        exit(-1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.device_count())
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']

    print(OmegaConf.to_yaml(cfg))
    validate_cfg(cfg)
    opt = options(cfg)
    print(opt)
    feature_extractor, model = load_model(cfg, opt)
    train_dataset, val_dataset = prepare_dataset(cfg, opt, feature_extractor)

    # Train
    print('Training...')
    wandb.init(project=cfg.exp.wandb.project,
               group=cfg.exp.wandb.group, name=cfg.exp.name)

    trainer_args = TrainingArguments(
        os.path.join(output_dir, cfg.exp.name),
        save_strategy='epoch',
        evaluation_strategy="epoch",
        learning_rate=cfg.exp.train.learning_rate,
        per_device_train_batch_size=int(
            cfg.exp.train.batch_size / torch.cuda.device_count()),
        per_device_eval_batch_size=16,
        num_train_epochs=cfg.exp.train.num_epochs,
        weight_decay=cfg.exp.train.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model='rmse' if opt.problem_type == 'regression' else 'accuracy',
        logging_strategy=cfg.exp.train.logging_strategy,
        logging_steps=cfg.exp.train.logging_steps,
        remove_unused_columns=False,
        report_to='wandb',
    )
    
    if cfg.exp.target == 'expression':
        trainer = WeightedLossTrainer(
            model,
            trainer_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=Collator(),
            compute_metrics=compute_accuracy,
            tokenizer=feature_extractor,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0002)],
        )
    elif cfg.exp.target == 'valence-arousal':
        trainer = KDEwMSETrainer(
            model,
            trainer_args,
            band_width=0.15,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=Collator(),
            compute_metrics=compute_rmse,
            tokenizer=feature_extractor,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0002)],
        )
    elif cfg.exp.target == 'arousal' or cfg.exp.target == 'valence':
        trainer = KDEwMSETrainer(
            model,
            trainer_args,
            band_width=0.15,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=Collator(),
            compute_metrics=compute_accuracy,
            tokenizer=feature_extractor,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0002)],
        )

    trainer.train()
    trainer.save_model(os.path.join(output_dir, '/model'))

    # Evaluate
    print('Evaluating...')
    evaluate(cfg.exp.data.images_root,
             cfg.exp.data.val_csv,
             cfg.exp.data.exclude_labels,
             cfg.exp.data.val_invalid_files,
             model,
             feature_extractor,
             20,
             device,
             output_dir,
             accuracy=True if opt.problem_type == 'single_label_classification' else False,
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
