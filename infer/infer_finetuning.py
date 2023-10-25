# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import torch
from datasets import Audio
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser, AutoConfig
from data_utils import train_info_args, NN_DataHelper, get_deepspeed_config
from aigc_zoo.model_zoo.asr_seq2seq.llm_model import MyTransformer


deep_config = get_deepspeed_config()


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(train_info_args, allow_extra_keys=True)

    dataHelper = NN_DataHelper(model_args)
    tokenizer, _, _,_= dataHelper.load_tokenizer_and_config()
    config = AutoConfig.from_pretrained('./best_ckpt')

    processor = dataHelper.processor
    config.forced_decoder_ids = None
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="chinese", task="transcribe")

    
    pl_model = MyTransformer(config=config, model_args=model_args,torch_dtype=config.torch_dtype,)

    # deepspeed 权重使用转换脚本命令
    # 一般根据时间排序选最新的权重文件夹
    # cd best_ckpt/last
    # python zero_to_fp32.py . ../last.ckpt

    train_weight = './best_ckpt/last.ckpt'
    pl_model.load_sft_weight(train_weight,strict=True)

    # 保存hf权重
    # config.save_pretrained('convert/')

    # 保存sft p-tuning-v2 权重
    #  pl_model.save_sft_weight('convert/pytorch_model_sft_ptv2.bin')

    # 保存sft权重
    # pl_model.save_sft_weight('convert/pytorch_model_sft.bin')

    model = pl_model.get_llm_model()

    model.eval().half().cuda()

    sample = Audio(sampling_rate=processor.feature_extractor.sampling_rate).decode_example({
        "path": "../assets/zh-CN_train_0/common_voice_zh-CN_18654294.mp3", "bytes": None
    })

    input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"],
                               return_tensors="pt").input_features

    input_features = input_features.half().to(model.device)
    # generate token ids
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids)

    print(transcription)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    print(transcription)