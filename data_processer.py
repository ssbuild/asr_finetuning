# @Time    : 2023/3/25 18:36
# @Author  : tk
import copy
import numpy as np
from datasets import Audio
from transformers import PreTrainedTokenizer


class TokenIdsMaker:
    @classmethod
    def process(cls, data_args,
                tokenizer: PreTrainedTokenizer,
                config,
                max_seq_length,
                feature_extractor,
                forward_attention_mask,
                examples):
        do_lower_case = data_args.data_custom["labels_do_lower_case"]
        max_input_length = data_args.max_duration_in_seconds * feature_extractor.sampling_rate
        min_input_length = data_args.min_duration_in_seconds * feature_extractor.sampling_rate
        sampling_rate = data_args.sampling_rate or feature_extractor.sampling_rate
        d = {}
        path,sentence = examples

        sample = Audio(sampling_rate=sampling_rate).decode_example({
            "path": path,"bytes": None
        })
        length = len(sample["array"])
        if length <= min_input_length and length > max_input_length:
            return None

        inputs = feature_extractor(
            sample["array"], sampling_rate=sample["sampling_rate"], return_attention_mask=forward_attention_mask
        )

        input_features = inputs["input_features"][0]
        d["shape"] = np.asarray(list(input_features.shape),dtype=np.int32)
        d["input_features"] = input_features.reshape(-1)
        if forward_attention_mask:
            d["attention_mask"] = inputs.get("attention_mask")[0]

        # process targets
        input_str = sentence.lower() if do_lower_case else sentence
        input_ids = tokenizer(input_str).input_ids
        labels = input_ids[:max_seq_length] if max_seq_length > 0 else input_ids
        d["labels"] = np.asarray(labels,dtype=np.int32)
        return d



