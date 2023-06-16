import torch
from transformers import Trainer

from SupContrast.losses import SupConLoss


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
