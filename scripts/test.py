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
import matplotlib.pyplot as plt
import wandb

from visualizer import CLS_tokens, plot_tokens_category, plot_tokens_continuous
from config import ContrastiveExpConfig, FinetuningExpConfig
from options import Options, options
from utils import exclude_id, try_finish_wandb

from evaluate import evaluate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='exp yaml file')
    parser.add_argument('model', help='model')
    parser.add_argument('output_dir', help='output directory')
    parser.add_argument('wandb_id', help='wandb run id')
    parser.add_argument('--accuracy', action='store_true', help='evaluate accuracy')
    parser.add_argument('--wandb_log', action='store_true', help='wandb logging')
    parser.add_argument('--wandb_resume', action='store_true', help='resume exsisting wandb run (require wandb_id)')

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
