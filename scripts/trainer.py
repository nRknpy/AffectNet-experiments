import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

from SupContrast.losses import SupConLoss
from KDEweightedMSE.losses import KDEWeightedMSESc

from losses import ContinuousSupConLoss


class AlternatingTrainer(Trainer):
    def __init__(self,
                 loss_weights = [1.0, 1.0],
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
        self.loss_fct1 = ContinuousSupConLoss()
        self.loss_fct2 = SupConLoss()
        self.loss_weights = loss_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs1, inputs2 = inputs
        labels1 = inputs1.get('labels')
        bsz1 = labels1.shape[0]
        labels2 = inputs2.get('labels')
        bsz2 = labels2.shape[0]
        
        outputs1 = model(pixel_values=inputs1.get('pixel_values'), output_hidden_states=True)
        features1 = outputs1.get('logits')
        features1 = F.normalize(features1, dim=1)
        f11, f12 = torch.split(features1, [bsz1, bsz1])
        features1 = torch.cat([f11.unsqueeze(1), f12.unsqueeze(1)], dim=1)
        loss1 = self.loss_fct1(features1, labels1)
        
        outputs2 = model(pixel_values=inputs2.get('pixel_values'), output_hidden_states=True)
        features2 = outputs2.get('logits')
        features2 = F.normalize(features2, dim=1)
        f21, f22 = torch.split(features2, [bsz2, bsz2])
        features2 = torch.cat([f21.unsqueeze(1), f22.unsqueeze(1)], dim=1)
        loss2 = self.loss_fct2(features2, labels2)
        
        loss = self.loss_weights[0] * loss1 + self.loss_weights[1] * loss2
        return (loss, [features1, features2]) if return_outputs else loss


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
                 model = None,
                 args = None,
                 data_collator = None,
                 train_dataset = None,
                 eval_dataset = None,
                 tokenizer = None,
                 model_init = None,
                 compute_metrics = None,
                 callbacks = None,
                 optimizers = (None, None),
                 preprocess_logits_for_metrics = None):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        d = train_dataset.df['expression'].value_counts().to_dict()
        label_samples_num = torch.tensor([d[i] for i in range(len(d))])
        label_ratio = label_samples_num / len(train_dataset)
        self.weight = (1 / label_ratio).clone().to(self.args.device, torch.float32)
        self.loss_fct = nn.CrossEntropyLoss(weight=self.weight)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss = self.loss_fct(logits.view(-1, len(self.weight)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class KDEwMSETrainer(Trainer):
    def __init__(self,
                 band_width = None,
                 model = None,
                 args = None,
                 data_collator = None,
                 train_dataset = None,
                 eval_dataset = None,
                 tokenizer = None,
                 model_init = None,
                 compute_metrics = None,
                 callbacks = None,
                 optimizers = (None, None),
                 preprocess_logits_for_metrics = None):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        
        data = train_dataset.df[['valence', 'arousal']]
        self.loss_fct = KDEWeightedMSESc(data=data, band_width=band_width, device=self.args.device, mode='divide', standardize=False)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss = self.loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
