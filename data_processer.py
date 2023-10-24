# @Time    : 2023/3/25 18:36
# @Author  : tk
import copy
from enum import Enum
import numpy as np
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
    def tunction(cls, tokenizer: PreTrainedTokenizer, config, sup, max_seq_length, examples):
        sptoken = [config.bos_token_id]
        ds = []

        path,sentence = examples
        a_ids = tokenizer.encode(text=build_template(q, prefix=prefix, history=examples[:sid]),
                                 add_special_tokens=False)
        b_ids = tokenizer.encode(text=a, add_special_tokens=False)
        while len(a_ids) + len(b_ids) > max_seq_length - len(sptoken) - 1:
            if len(b_ids) > len(a_ids):
                b_ids.pop(-1)
            else:
                a_ids.pop(0)
        b_ids += [config.eos_token_id]
        input_ids = a_ids + b_ids
        labels = copy.deepcopy(input_ids) if not sup else [-100] * len(a_ids) + copy.deepcopy(b_ids)
        input_ids = sptoken + input_ids
        labels = sptoken + labels if not sup else [-100] * len(sptoken) + labels
        assert len(input_ids) <= max_seq_length
        ds.append(cls.final(tokenizer, input_ids, labels, max_seq_length))

        return ds



