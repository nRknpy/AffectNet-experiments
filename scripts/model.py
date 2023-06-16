from transformers import ViTFeatureExtractor, ViTForImageClassification

from config import ContrastiveExpConfig, FinetuningExpConfig
from options import Options


def load_model(cfg: ContrastiveExpConfig | FinetuningExpConfig, opt: Options):
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        cfg.exp.model.model_ckpt)
    if cfg.exp.type == 'contrastive':
        model = ViTForImageClassification.from_pretrained(cfg.exp.model.model_ckpt,
                                                          num_labels=opt.num_labels,
                                                          problem_type=opt.problem_type)
    elif cfg.exp.type == 'finetuning':
        model = ViTForImageClassification.from_pretrained(cfg.exp.model.model_ckpt,
                                                          num_labels=opt.num_labels,
                                                          id2label=opt.id2label,
                                                          label2id=opt.label2id,
                                                          problem_type=opt.problem_type,
                                                          ignore_mismatched_sizes=True)

    return feature_extractor, model
