# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer_lora_finetuning
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import os
import torch
from datasets import Audio
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser,AutoConfig
from data_utils import train_info_args, NN_DataHelper,global_args
from aigc_zoo.model_zoo.asr_seq2seq.llm_model import MyTransformer,PetlArguments,PromptArguments



if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(train_info_args, allow_extra_keys=True)


    dataHelper = NN_DataHelper(model_args)
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config()

    # 一般根据时间排序选最新的权重文件夹
    weight_dir = '../scripts/best_ckpt'
    lora_weight_dir = os.path.join(weight_dir, "last")

    config = AutoConfig.from_pretrained(weight_dir)
    lora_args = PetlArguments.from_pretrained(lora_weight_dir)

    processor = dataHelper.processor
    config.forced_decoder_ids = None
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="chinese", task="transcribe")

    assert lora_args.inference_mode == True

    new_num_tokens = config.vocab_size
    if config.task_specific_params is not None and config.task_specific_params.get('vocab_size',None) is not None:
        config.vocab_size = config.task_specific_params['vocab_size']

    pl_model = MyTransformer(config=config, model_args=model_args,
                             lora_args=lora_args,
                             torch_dtype=config.torch_dtype,
                             new_num_tokens=new_num_tokens,
                             # load_in_8bit=global_args["load_in_8bit"],
                             # # device_map="auto",
                             # device_map = {"":0} # 第一块卡
                             )

    # 加载lora权重
    pl_model.load_sft_weight(lora_weight_dir)

    pl_model.eval().half().cuda()

    enable_merge_weight = False

    if enable_merge_weight:
        # 合并lora 权重 保存
        pl_model.save_sft_weight(os.path.join(lora_weight_dir, 'pytorch_model_merge.bin'), merge_lora_weight=True)
    else:
        model = pl_model.get_llm_model()

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