import importlib
import copy
from .dataset_config import *
from .model_config import *


ModelConfig = {
    "PRETUNE": model_config_pretune,
    "FINETUNE": model_config_finetune,
}


## automatically add additional model_config

module = importlib.import_module('ChatUniVi.config.model_config')
dict_names = [x for x in dir(module) if 
              isinstance(getattr(module, x), dict) and \
              x.startswith('model_config') and \
              x not in ['model_config_pretune', 'model_config_finetune']
              ]
for dict_name in dict_names:
    if any(x in dict_name for x in ['pretune', 'fintune']):
        raise ValueError('should not contain pretune/finetune since will make one for each.')
    for finetune_type in ['pretune', 'finetune']:
        d = copy.deepcopy(getattr(module, dict_name))
        if finetune_type == 'pretune':
            d['use_cluster'] = d.get('use_cluster', True)
            d['freeze'] = d.get('freeze', False)
            d['vision_tune'] = d.get('vision_tune', False)
        elif finetune_type == 'finetune':
            d['use_cluster'] = d.get('use_cluster', True)
            d['freeze'] = d.get('freeze', False)
            d['mm_tune'] = d.get('mm_tune', True)
            d['vision_tune'] = d.get('vision_tune', False)
        ModelConfig.update({finetune_type + '_' + dict_name.split('model_config_')[1]: d})



DataConfig = {
    "Pretrain": [Pretrain, COCO_CAP, COCO_REG, COCO_REC],
    "SQA": [SQA],
    "FINETUNE": [VIT, MIMIC_imageonly, VIDEO],
    "Pretrainv1.5": [Pretrain, Pretrain_valley_llava],
    "FINETUNEv1.5": [VIT, VIDEO, LLaVA],
    'valley703k+llavalcs558k': [Pretrain_valley_llava],
    'valleyllavalcs300k': [Pretrain_valley_llava_300k],
    'vcg100k+llava665k': [VIDEO, LLaVA],
    'vcg100k+llava200k': [VIDEO, LLaVA200k],
    'vit150k+vcg100k+llava665k': [VIT, VIDEO, LLaVA],
}