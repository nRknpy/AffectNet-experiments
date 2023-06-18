import torch
import torch.nn as nn
from transformers import Trainer

from SupContrast.losses import SupConLoss
from KDEweightedMSE.losses import KDEWeightedMSESc


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
        bsz = labels.shape[0]
        outputs = model(pixel_values=inputs.get(
            'pixel_values'), output_hidden_states=True)
        features = outputs.get('logits')
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
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = nn.CrossEntropyLoss(weight=self.weight)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
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
