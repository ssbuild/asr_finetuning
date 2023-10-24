## update information
   - [deep_training](https://github.com/ssbuild/deep_training)

```text
    10-24 initial asr seq2seq
```
   

## install
  - pip install -U -r requirements.txt
  - 如果无法安装， 可以切换官方源 pip install -i https://pypi.org/simple -U -r requirements.txt



## weigtht select one is suitable for you
支持且不限于以下权重    
- [whisper-large-v2](https://huggingface.co/openai/whisper-large-v2)
- [whisper-large](https://huggingface.co/openai/whisper-large)
- [whisper-base](https://huggingface.co/openai/whisper-base)
- [whisper-small](https://huggingface.co/openai/whisper-small)
- [whisper-tiny](https://huggingface.co/openai/whisper-tiny)

## data sample
- open_data https://github.com/ssbuild/open_data
- common_voice_11_0 https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0
- common_voice https://huggingface.co/datasets/common_voice
   
单条数据示例
```text
p prefix  optional
q question optional
a answer   must

```

```json
{
  "path": "../zh-CN_train_0/common_voice_zh-CN_33211332.mp3",
  "sentence": "性喜温暖润湿气候且耐寒。",
  "locale": "zh-CN"
}
```



## infer
    # infer_finetuning.py 推理微调模型
    # infer_lora_finetuning.py 推理微调模型
     python infer_finetuning.py



## training
```text
    # 制作数据
    cd scripts
    bash train_full.sh -m dataset 
    or
    bash train_lora.sh -m dataset 
    or
    bash train_ptv2.sh -m dataset 
    
    注: num_process_worker 为多进程制作数据 ， 如果数据量较大 ， 适当调大至cpu数量
    dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False, shuffle=True,mode='train',num_process_worker=0)
    
    # 全参数训练 
        bash train_full.sh -m train
        
    # lora adalora ia3 
        bash train_lora.sh -m train
        
    # ptv2
        bash train_ptv2.sh -m train
```
   
## 训练参数
[训练参数](args.MD)

## 友情链接

- [pytorch-task-example](https://github.com/ssbuild/pytorch-task-example)
- [tf-task-example](https://github.com/ssbuild/tf-task-example)
- [chatmoss_finetuning](https://github.com/ssbuild/chatmoss_finetuning)
- [chatglm_finetuning](https://github.com/ssbuild/chatglm_finetuning)
- [t5_finetuning](https://github.com/ssbuild/t5_finetuning)
- [llm_finetuning](https://github.com/ssbuild/llm_finetuning)
- [llm_rlhf](https://github.com/ssbuild/llm_rlhf)
- [chatglm_rlhf](https://github.com/ssbuild/chatglm_rlhf)
- [t5_rlhf](https://github.com/ssbuild/t5_rlhf)
- [rwkv_finetuning](https://github.com/ssbuild/rwkv_finetuning)
- [baichuan_finetuning](https://github.com/ssbuild/baichuan_finetuning)

## 
    纯粹而干净的代码

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ssbuild/asr_seq2seq_finetuning&type=Date)](https://star-history.com/#ssbuild/asr_seq2seq_finetuning&Date)