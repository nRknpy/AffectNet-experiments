from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from SupContrast.losses import SupConLoss


class AffeAndLangSupConLoss(nn.Module):
    def __init__(self,
                 affe_feature_temp: float = 0.07,
                 affe_label_temp: float = 0.07,
                 affe_feature_sim: Literal['dot', 'l2norm'] = 'dot',
                 affe_label_sim: Literal['dot', 'l2norm'] = 'l2norm',
                 lang_feature_temp: float = 0.07,
                 ):
        super(AffeAndLangSupConLoss, self).__init__()
        self.affe_feature_temp = affe_feature_temp
        self.affe_label_temp = affe_label_temp
        self.affe_feature_sim = affe_feature_sim
        self.affe_label_sim = affe_label_sim
        self.lang_feature_temp = lang_feature_temp

        if not self.affe_feature_sim in ('dot', 'l2norm'):
            raise ValueError('Unknown feature similarity mode: {}'.format(
                self.affe_feature_sim))
        if not self.affe_label_sim in ('dot', 'l2norm'):
            raise ValueError('Unknown label similarity mode: {}'.format(
                self.affe_label_sim))

        self.affe_loss_fct = ContinuousSupConLoss(self.affe_feature_temp,
                                                  self.affe_label_temp,
                                                  self.affe_feature_sim,
                                                  self.affe_label_sim)
        self.lang_loss_fct = SupConLoss(self.lang_feature_temp)

    def forward(self, features, affe_labels, lang_labels, affe_rate):
        affe_loss = self.affe_loss_fct(features, affe_labels)
        lang_loss = self.lang_loss_fct(features, lang_labels)
        loss = affe_rate * affe_loss + (1 - affe_rate) * lang_loss
        return loss


class ContinuousSupConLoss(nn.Module):
    def __init__(self,
                 feature_temperature: float = 0.07,
                 label_temperature: float = 0.07,
                 feature_similarity_mode: Literal['dot', 'l2norm'] = 'dot',
                 label_similarity_mode: Literal['dot', 'l2norm'] = 'l2norm'):
        super(ContinuousSupConLoss, self).__init__()
        self.label_temperature = label_temperature
        self.feature_temperature = feature_temperature
        self.f_similarity_mode = feature_similarity_mode
        self.l_similarity_mode = label_similarity_mode

        if not self.f_similarity_mode in ('dot', 'l2norm'):
            raise ValueError('Unknown feature similarity mode: {}'.format(
                self.f_similarity_mode))
        if not self.l_similarity_mode in ('dot', 'l2norm'):
            raise ValueError('Unknown label similarity mode: {}'.format(
                self.l_similarity_mode))

    def dot_similarity(self, anchor, contrast, temp, submax=True):
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor, contrast.T),
            temp)
        if submax:
            _max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            anchor_dot_contrast = anchor_dot_contrast - _max.detach()
        return anchor_dot_contrast

    def l2norm_similarity(self, anchor, contrast, temp, submax=True):
        dist_anchor_contrast = -torch.div(
            torch.cdist(anchor, contrast, p=2),
            temp
        )
        if submax:
            _max, _ = torch.max(dist_anchor_contrast, dim=1, keepdim=True)
            dist_anchor_contrast = dist_anchor_contrast - _max.detach()
        return dist_anchor_contrast

    def forward(self, features, labels):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz, label_dim, ...].
        Returns:
            A loss scalar.
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        if len(labels.shape) != 2:
            labels = labels.view(labels.shape[0], -1)

        batch_size = features.shape[0]
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        contrast_label = labels.repeat(contrast_count, 1)
        anchor_label = contrast_label

        if self.f_similarity_mode == 'dot':
            logits = self.dot_similarity(
                anchor_feature, contrast_feature, self.feature_temperature)
        elif self.f_similarity_mode == 'l2norm':
            logits = self.l2norm_similarity(
                anchor_feature, contrast_feature, self.feature_temperature)

        if self.l_similarity_mode == 'dot':
            weights = self.dot_similarity(
                anchor_label, contrast_label, self.label_temperature)
        elif self.l_similarity_mode == 'l2norm':
            weights = self.l2norm_similarity(
                anchor_label, contrast_label, self.label_temperature)

        i_mask = torch.scatter(
            torch.ones(batch_size * anchor_count,
                       batch_size * contrast_count).to(device),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        exp_logits = torch.exp(logits) * i_mask
        logits_log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        exp_weights = torch.exp(weights) * i_mask
        weights_log_prob = weights - \
            torch.log(exp_weights.sum(1, keepdim=True))
        weights_prob = torch.exp(weights_log_prob)

        log_prob = torch.mul(weights_prob, logits_log_prob)

        mean_log_prob = log_prob.sum(1) / i_mask.sum(1)

        # loss
        loss = - mean_log_prob
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
