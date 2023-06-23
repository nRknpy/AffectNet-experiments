from typing import List, Tuple, Dict, Any
import torch
import numpy as np
import pandas as pd
import os
from omegaconf import OmegaConf
import argparse
from hydra.experimental import compose, initialize_config_dir
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchaffectnet import AffectNetDatasetForSupCon, AffectNetDataset
from torchaffectnet.collators import Collator
from torchaffectnet.const import ID2EXPRESSION
from dataset import AffectNetDatasetForSupConWithValence
from transformers import ViTFeatureExtractor, ViTForImageClassification, Trainer, TrainingArguments
from datasets import load_metric
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import wandb
from PIL import Image

from visualizer import CLS_tokens, plot_tokens_category, plot_tokens_continuous
from config import ContrastiveExpConfig, FinetuningExpConfig
from options import Options, options
from utils import exclude_id, try_finish_wandb


def prepare_datasets(images_root: str, csvfile: str, exclude_labels: List[int], invalid_files: List[str], feature_extractor: Tuple[ViTFeatureExtractor, Dict[str, Any]] | ViTFeatureExtractor):
    normalize = Normalize(mean=feature_extractor.image_mean,
                          std=feature_extractor.image_std)
    transform = Compose([
        Resize(tuple(feature_extractor.size.values())),
        ToTensor(),
        normalize,
    ])

    category_dataset = AffectNetDataset(csvfile,
                                        images_root,
                                        exclude_label=exclude_labels,
                                        invalid_files=invalid_files,
                                        transform=transform,
                                        mode='classification')
    cat_valence_dataset = AffectNetDatasetForSupConWithValence(csvfile,
                                                               images_root,
                                                               exclude_label=exclude_labels,
                                                               invalid_files=invalid_files,
                                                               transform1=transform,
                                                               transform2=transform)
    valence_dataset = AffectNetDataset(csvfile,
                                       images_root,
                                       exclude_label=exclude_labels,
                                       invalid_files=invalid_files,
                                       transform=transform,
                                       mode='valence')
    arousal_dataset = AffectNetDataset(csvfile,
                                       images_root,
                                       exclude_label=exclude_labels,
                                       invalid_files=invalid_files,
                                       transform=transform,
                                       mode='arousal')

    return [
        category_dataset,
        cat_valence_dataset,
        valence_dataset,
        arousal_dataset,
    ]


output_names = [
    'emotion.png',
    'categorical_valence.png',
    'valence.png',
    'arousal.png'
]


acc_met = load_metric("accuracy")


def compute_accuracy(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return acc_met.compute(predictions=predictions, references=labels)


def compute_rmse(eval_pred):
    preds, targets = eval_pred
    rmse = mean_squared_error(targets, preds, squared=False)
    return {'rmse': rmse}


def evaluate(images_root,
             csvfile,
             exclude_labels,
             invalid_files,
             model,
             feature_extractor,
             umap_n_neighbors,
             device,
             output_dir,
             random_seed,
             accuracy=False,
             wandb_log=False,
             wandb_resume=False,
             wandb_proj=None,
             wandb_group=None,
             wandb_name=None,
             wandb_id=None,
             after_train=False):
    if wandb_log:
        if wandb_resume:
            if wandb_id == None:
                print('To resume wandb run, it need wandb_id.')
                exit(-1)
            wandb.init(project=wandb_proj, group=wandb_group,
                       name=wandb_name, id=wandb_id, resume='must')
        else:
            if not after_train:
                wandb.init(project=wandb_proj,
                           group=wandb_group, name=wandb_name)

    model = model.to(device)
    datasets = prepare_datasets(
        images_root, csvfile, exclude_labels, invalid_files, feature_extractor)

    cat_id2label, cat_label2id = exclude_id(exclude_labels)

    if accuracy:
        args = TrainingArguments(
            'evaluate',
            remove_unused_columns=False,
            metric_for_best_model='accuracy',
            report_to='none'
        )
        trainer = Trainer(
            model=model,
            args=args,
            data_collator=Collator(),
            compute_metrics=compute_accuracy,
            tokenizer=feature_extractor,
        )
        test_outputs = trainer.predict(datasets[0])
        print('='*30)
        print(test_outputs.metrics)
        print('='*30)

        y_true = test_outputs.label_ids
        y_pred = test_outputs.predictions.argmax(1)

        labels = list(cat_label2id.keys())
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(12, 12))
        conmat = disp.plot(ax=ax)
        conmat.figure_.savefig(os.path.join(
            output_dir, 'confusion_matrix.png'))
        if wandb_log:
            im = Image.open(os.path.join(output_dir, 'confusion_matrix.png'))
            wandb.log({'confusion_matrix.png': wandb.Image(im)})

    id2labels = [
        cat_id2label,
        {
            0: 'valence < -0.5',
            1: '-0.5 <= valence <= 0.5',
            2: '0.5 < valence',
        }
    ]

    for i, dataset in enumerate(datasets[:2]):
        tokens, targets = CLS_tokens(model, dataset, device)
        fig, legend = plot_tokens_category(
            tokens, targets, umap_n_neighbors, id2labels[i], random_seed)
        fig.savefig(os.path.join(output_dir, output_names[i]), bbox_extra_artists=[
                    legend], bbox_inches='tight')
        fig.add_artist(legend)
        if wandb_log:
            im = Image.open(os.path.join(output_dir, output_names[i]))
            wandb.log({output_names[i]: wandb.Image(im)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='exp yaml file')
    parser.add_argument('model', help='model')
    parser.add_argument('output_dir', help='output directory')
    parser.add_argument('wandb_id', help='wandb run id')
    parser.add_argument('--accuracy', action='store_true',
                        help='evaluate accuracy')
    parser.add_argument('--wandb_log', action='store_true',
                        help='wandb logging')
    parser.add_argument('--wandb_resume', action='store_true',
                        help='resume exsisting wandb run (require wandb_id)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.config):
        print(f"Can not find file: {args.config}.")
        exit(-1)

    with initialize_config_dir(config_dir=os.path.join(os.path.dirname(__file__), '..')):
        cfg: ContrastiveExpConfig | FinetuningExpConfig = compose(
            config_name=args.config)
    cfg = OmegaConf.to_object(cfg)
    while not 'name' in cfg.keys():
        print(cfg.keys())
        cfg = cfg[list(cfg.keys())[0]]
    print(cfg)

    model = ViTForImageClassification.from_pretrained(args.model)
    feature_extractor = ViTFeatureExtractor.from_pretrained(args.model)

    outputs = evaluate(cfg['data']['images_root'],
                       cfg['data']['val_csv'],
                       cfg['data']['exclude_labels'],
                       cfg['data']['val_invalid_files'],
                       model,
                       feature_extractor,
                       20,
                       device, args.output_dir,
                       cfg['random_seed'],
                       args.accuracy,
                       wandb_log=args.wandb_log,
                       wandb_resume=args.wandb_resume,
                       wandb_proj=cfg['wandb']['project'],
                       wandb_group=cfg['wandb']['group'],
                       wandb_name=cfg['name'],
                       wandb_id=args.wandb_id,
                       after_train=False)
    if args.wandb_log:
        # wandb.finish()
        try_finish_wandb()
