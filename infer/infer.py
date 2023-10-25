# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer
import os
import sys



sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import torch
from datasets import Audio
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser
from data_utils import train_info_args, NN_DataHelper, get_deepspeed_config
from aigc_zoo.model_zoo.asr_seq2seq.llm_model import MyTransformer


deep_config = get_deepspeed_config()


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments,))
    (model_args,)  = parser.parse_dict(train_info_args, allow_extra_keys=True)

    dataHelper = NN_DataHelper(model_args)
    tokenizer, config, _,_= dataHelper.load_tokenizer_and_config()
    processor = dataHelper.processor


    pl_model = MyTransformer(config=config, model_args=model_args,torch_dtype=config.torch_dtype)
    model = pl_model.get_llm_model()
    model = model.eval()
    model.half().cuda()

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
    # ['<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>他后来捧去打他两个儿光<|endoftext|>']
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    print(transcription)
    # ['他后来捧去打他两个儿光']
