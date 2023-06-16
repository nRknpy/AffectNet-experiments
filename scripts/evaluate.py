from typing import List, Tuple, Dict, Any
import torch
import numpy as np
import pandas as pd
import os
from omegaconf import OmegaConf
import argparse
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchaffectnet import AffectNetDatasetForSupCon, AffectNetDataset
from torchaffectnet.collators import Collator
from torchaffectnet.const import ID2EXPRESSION
from dataset import AffectNetDatasetForSupConWithValence
from transformers import ViTFeatureExtractor, ViTForImageClassification, Trainer, TrainingArguments
from datasets import load_metric
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from visualizer import CLS_tokens, plot_tokens_category, plot_tokens_continuous
from config import ContrastiveExpConfig, FinetuningExpConfig
from options import Options, options


def prepare_datasets(images_root: str, csvfile: str, exclude_labels: List[int], feature_extractor: Tuple[ViTFeatureExtractor, Dict[str, Any]] | ViTFeatureExtractor):
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
                                        transform=transform,
                                        mode='classification')
    cat_valence_dataset = AffectNetDatasetForSupConWithValence(csvfile,
                                                               images_root,
                                                               exclude_label=exclude_labels,
                                                               transform=transform)
    valence_dataset = AffectNetDataset(csvfile,
                                       images_root,
                                       exclude_label=exclude_labels,
                                       transform=transform,
                                       mode='valence')
    arousal_dataset = AffectNetDataset(csvfile,
                                       images_root,
                                       exclude_label=exclude_labels,
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


def evaluate(cfg: ContrastiveExpConfig | FinetuningExpConfig, opt: Options, model, feature_extractor, umap_n_neighbors, device, output_dir, accuracy=False):
    datasets = prepare_datasets(
        cfg.exp.data.images_root, cfg.exp.data.val_csv, cfg.exp.data.exclude_labels, feature_extractor)

    cat_id2label = ID2EXPRESSION
    for label in cfg.exp.data.exclude_labels:
        del cat_id2label[label]
    cat_label2id = {v: k for k, v in cat_id2label.items()}

    output = {}
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
        output['confusion_matrix.png'] = conmat

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
        fig = plot_tokens_category(
            tokens, targets, umap_n_neighbors, id2labels[i])
        output[output_names[i]] = fig

    for name, fig in output.values():
        fig.savefig(os.path.join(output_dir, name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='yaml file')
    parser.add_argument('model', help='model')
    parser.add_argument('output_dir', help='output directory')
    parser.add_argument('--accuracy', action='store_true')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg: ContrastiveExpConfig | FinetuningExpConfig = OmegaConf.load(
        args.config)
    opt = options(cfg)
    model = ViTForImageClassification.from_pretrained(args.model)
    feature_extractor = ViTFeatureExtractor.from_pretrained(args.model)

    outputs = evaluate(cfg, opt, model, feature_extractor,
                       20, device, args.output_dir, args.accuracy)
