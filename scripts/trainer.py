import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from transformers import Trainer

from SupContrast.losses import SupConLoss
from KDEweightedMSE.losses import KDEWeightedMSESc

from losses import ContinuousSupConLoss, AffeAndLangSupConLoss


class AffeAndLangTrainer(Trainer):
    def __init__(self,
                 model=None,
                 args=None,
                 data_collator=None,
                 train_dataset=None,
                 eval_dataset=None,
                 tokenizer=None,
                 model_init=None,
                 compute_metrics=None,
                 callbacks=None,
                 optimizers=(None, None),
                 preprocess_logits_for_metrics=None):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer,
                         model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        self.loss_fct = AffeAndLangSupConLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        current_epoch = self.state.epoch
        affe_rate = 0.5 * \
            (1 + math.cos(math.pi * current_epoch / self.args.num_train_epochs))

        labels = inputs.get('labels')
        affe_labels = labels[:, :2]
        lang_labels = labels[:, -1].int()
        bsz = labels.shape[0]
        outputs = model(pixel_values=inputs.get(
            'pixel_values'), output_hidden_states=True)
        features = outputs.get('logits')
        features = F.normalize(features, dim=1)
        f1, f2 = torch.split(features, [bsz, bsz])
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = self.loss_fct(features, affe_labels, lang_labels, affe_rate)
        return (loss, features) if return_outputs else loss


class AlternatingTrainer(Trainer):
    def __init__(self,
                 model=None,
                 args=None,
                 data_collator=None,
                 train_dataset=None,
                 eval_dataset=None,
                 tokenizer=None,
                 model_init=None,
                 compute_metrics=None,
                 callbacks=None,
                 optimizers=(None, None),
                 preprocess_logits_for_metrics=None):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer,
                         model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        self.loss_fcts = [ContinuousSupConLoss(),
                          SupConLoss()]

    def compute_loss(self, model, inputs, return_outputs=False):
        dataset_id = inputs.get('dataset_id')
        labels = inputs.get('labels')
        outputs = model(pixel_values=inputs.get(
            'pixel_values'), output_hidden_states=True)
        features = outputs.get('logits')
        if labels is not None:
            bsz = labels.shape[0]
        else:
            bsz = features.shape[0] // 2
        features = F.normalize(features, dim=1)
        f1, f2 = torch.split(features, [bsz, bsz])
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = self.loss_fcts[dataset_id](features, labels)
        return (loss, features) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="training")

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


class ContinuousSupConTrainer(Trainer):
    def __init__(self,
                 model=None,
                 args=None,
                 data_collator=None,
                 train_dataset=None,
                 eval_dataset=None,
                 tokenizer=None,
                 model_init=None,
                 compute_metrics=None,
                 callbacks=None,
                 optimizers=(None, None),
                 preprocess_logits_for_metrics=None):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer,
                         model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        self.loss_fct = ContinuousSupConLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        bsz = labels.shape[0]
        outputs = model(pixel_values=inputs.get(
            'pixel_values'), output_hidden_states=True)
        features = outputs.get('logits')
        features = F.normalize(features, dim=1)
        f1, f2 = torch.split(features, [bsz, bsz])
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = self.loss_fct(features, labels)
        return (loss, features) if return_outputs else loss


class SupConTrainer(Trainer):
    def __init__(self,
                 model=None,
                 args=None,
                 data_collator=None,
                 train_dataset=None,
                 eval_dataset=None,
                 tokenizer=None,
                 model_init=None,
                 compute_metrics=None,
                 callbacks=None,
                 optimizers=(None, None),
                 preprocess_logits_for_metrics=None):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer,
                         model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        self.loss_fct = SupConLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(pixel_values=inputs.get(
            'pixel_values'), output_hidden_states=True)
        features = outputs.get('logits')
        if labels is not None:
            bsz = labels.shape[0]
        else:
            bsz = features.shape[0] // 2
        features = F.normalize(features, dim=1)
        f1, f2 = torch.split(features, [bsz, bsz])
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = self.loss_fct(features, labels)
        return (loss, features) if return_outputs else loss


class WeightedLossTrainer(Trainer):
    def __init__(self,
                 model=None,
                 args=None,
                 data_collator=None,
                 train_dataset=None,
                 eval_dataset=None,
                 tokenizer=None,
                 model_init=None,
                 compute_metrics=None,
                 callbacks=None,
                 optimizers=(None, None),
                 preprocess_logits_for_metrics=None):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer,
                         model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        d = train_dataset.df['expression'].value_counts().to_dict()
        label_samples_num = torch.tensor([d[i] for i in range(len(d))])
        label_ratio = label_samples_num / len(train_dataset)
        self.weight = (
            1 / label_ratio).clone().to(self.args.device, torch.float32)
        self.loss_fct = nn.CrossEntropyLoss(weight=self.weight)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss = self.loss_fct(
            logits.view(-1, len(self.weight)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class KDEwMSETrainer(Trainer):
    def __init__(self,
                 band_width=None,
                 model=None,
                 args=None,
                 data_collator=None,
                 train_dataset=None,
                 eval_dataset=None,
                 tokenizer=None,
                 model_init=None,
                 compute_metrics=None,
                 callbacks=None,
                 optimizers=(None, None),
                 preprocess_logits_for_metrics=None):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer,
                         model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)

        data = train_dataset.df[['valence', 'arousal']]
        self.loss_fct = KDEWeightedMSESc(
            data=data, band_width=band_width, device=self.args.device, mode='divide', standardize=False)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss = self.loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
