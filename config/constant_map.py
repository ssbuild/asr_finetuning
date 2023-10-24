# -*- coding: utf-8 -*-
# @Time:  23:20
# @Author: tk
# @File：model_maps
from aigc_zoo.constants.define import (TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING)

__all__ = [
    "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING",
    "MODELS_MAP"
]

MODELS_MAP = {
    'whisper-large-v2': {
        'model_type': 'whisper',
        'model_name_or_path': '/data/nlp/pre_models/torch/whisper/whisper-large-v2',
        'config_name': '/data/nlp/pre_models/torch/whisper/whisper-large-v2/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/whisper/whisper-large-v2',
    },
    'whisper-large': {
        'model_type': 'whisper',
        'model_name_or_path': '/data/nlp/pre_models/torch/whisper/whisper-large',
        'config_name': '/data/nlp/pre_models/torch/whisper/whisper-large/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/whisper/whisper-large',
    },

    'whisper-base': {
        'model_type': 'whisper',
        'model_name_or_path': '/data/nlp/pre_models/torch/whisper/whisper-base',
        'config_name': '/data/nlp/pre_models/torch/whisper/whisper-base/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/whisper/whisper-base',
    },

    'whisper-small': {
        'model_type': 'whisper',
        'model_name_or_path': '/data/nlp/pre_models/torch/whisper/whisper-small',
        'config_name': '/data/nlp/pre_models/torch/whisper/whisper-small/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/whisper/whisper-small',
    },

}


# 按需修改
# TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING




