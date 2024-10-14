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


DataConfig = {}

## chatunivi / video-llava
DataConfig.update({
    # pretrain (image only)
    'llavalcs': [CC3M_595K],
    # pretrain (video only)
    'vcg100k': [pretrain_valley_300k],
    # pretrain (mix)
    'valley703k+llavalcs558k': [pretrain_valley_llava],
    'valleyllavalcs300k': [pretrain_valley_llava_300k],
    # sft (image only)
    'llava': [SFT_LLaVA],
    'llava200k': [SFT_LLaVA_200k],
    # sft (video only)
    'vcg100k': [CONV_VideoChatGPT],
    # sft (mix)
    'vcg100k+llava665k': [CONV_VideoChatGPT, SFT_LLaVA],
    'vcg100k+llava200k': [CONV_VideoChatGPT, SFT_LLaVA_200k],
})

## videogpt+
DataConfig.update({
    'sft+videogptplus+all': [CONV_VideoChatGPT, VCG_HUMAN, VCG_PLUS_112K, CAPTION_VIDEOCHAT, CLASSIFICATION_K710, CLASSIFICATION_SSV2, CONV_VideoChat1, REASONING_NExTQA, REASONING_CLEVRER_QA, REASONING_CLEVRER_MC, VQA_WEBVID_QA],
    "sft+videogptplus+vcgbench": [CONV_VideoChatGPT, VCG_HUMAN, VCG_PLUS_112K, CAPTION_VIDEOCHAT, CONV_VideoChat1, VQA_WEBVID_QA],
    "sft+videogptplus+mvbench": [CLASSIFICATION_K710, CLASSIFICATION_SSV2, CONV_VideoChatGPT, REASONING_NExTQA, REASONING_CLEVRER_QA, REASONING_CLEVRER_MC, VQA_WEBVID_QA],
})
