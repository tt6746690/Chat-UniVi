from .dataset_config import *
from .model_config import *


ModelConfig = {
    "PRETUNE": model_config_pretune,
    "FINETUNE": model_config_finetune,
}


DataConfig = {
    "Pretrain": [Pretrain, COCO_CAP, COCO_REG, COCO_REC],
    "SQA": [SQA],
    "FINETUNE": [VIT, MIMIC_imageonly, VIDEO],
    "Pretrainv1.5": [Pretrain, Pretrain_valley_llava],
    "FINETUNEv1.5": [VIT, VIDEO, LLaVA],
    'valley703k+llavalcs558k': [Pretrain_valley_llava],
    'vcg100k+llava665k': [VIDEO, LLaVA],
    'vcg100k+llava200k': [VIDEO, LLaVA200k],
    'vit150k+vcg100k+llava665k': [VIT, VIDEO, LLaVA],
}