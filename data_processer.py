# @Time    : 2023/3/25 18:36
# @Author  : tk
import copy
from enum import Enum
import numpy as np
from datasets import Audio
from transformers import PreTrainedTokenizer

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

class DataStrategy(Enum):
    tunction = 1






class TokenIdsMaker:
    @classmethod
    def final(cls, tokenizer, input_ids, labels, max_seq_length):
        seqlen = np.asarray(len(input_ids), dtype=np.int32)
        pad_len = max_seq_length - seqlen
        input_ids = np.asarray(input_ids, dtype=np.int32)
        attention_mask = np.asarray([1] * len(input_ids), dtype=np.int32)
        labels = np.asarray(labels, dtype=np.int32)
        if pad_len:
            pad_val = tokenizer.eos_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))
            labels = np.pad(labels, (0, pad_len), 'constant', constant_values=(-100, -100))
        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'seqlen': seqlen
        }
        return d
    @classmethod
    def tunction(cls, data_args,tokenizer: PreTrainedTokenizer, config, sup, max_seq_length,feature_extractor,sampling_rate,do_lower_case,forward_attention_mask, examples):
        max_input_length = data_args.max_duration_in_seconds * feature_extractor.sampling_rate
        min_input_length = data_args.min_duration_in_seconds * feature_extractor.sampling_rate
        sampling_rate = data_args.sampling_rate
        d = {}
        path,sentence = examples

        sample = Audio(sampling_rate=sampling_rate).decode_example({
            "path": path,"bytes": None
        })
        length = len(sample["array"])
        if length > min_input_length and length < max_input_length:
            return None

        inputs = feature_extractor(
            sample["array"], sampling_rate=sample["sampling_rate"], return_attention_mask=forward_attention_mask
        )

        d["input_features"] = inputs["input_features"]
        if forward_attention_mask:
            d["attention_mask"] = inputs.get("attention_mask")[0]

        # process targets
        input_str = sentence.lower() if do_lower_case else sentence
        input_ids = tokenizer(input_str).input_ids
        labels = input_ids[:max_seq_length] if max_seq_length > 0 else input_ids
        d["labels"] = np.asarray(labels,dtype=np.int32)
        return d



