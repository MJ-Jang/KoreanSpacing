import numpy as np
import random

from torch.utils.data import Dataset


class SpacingDataset(Dataset):
    def __init__(self, data, tok, max_len):
        self.tok = tok
        self.max_len = max_len
        self.data = data
        self.pad_id = tok.piece_to_id('[PAD]')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # tgt
        tgt = self.data[idx]

        # sample src
        text_wo_space = tgt.replace(' ', '')
        n_space = random.sample(range(tgt.count(' ') + 1), 1)[0]
        sampled = random.sample(range(len(text_wo_space)), n_space)

        src = [s + ' ' if i in sampled else s for i, s in enumerate(text_wo_space)]
        src = ''.join(src)

        tgt_id, tgt_len = self._tokenize(self.tok, tgt)
        src_id, src_len = self._tokenize(self.tok, src)

        if src_len < self.max_len:
            src_id = src_id + [self.pad_id] * (self.max_len - src_len)
        elif src_len > self.max_len:
            src_id = src_id[:self.max_len]

        if tgt_len < self.max_len:
            tgt_id = tgt_id + [self.pad_id] * (self.max_len - tgt_len)
        elif tgt_len > self.max_len:
            tgt_id = tgt_id[:self.max_len]
        return np.array(src_id), np.array(tgt_id)

    def _tokenize(self, tokenizer, sent):
        tokens = tokenizer.encode_as_ids(sent)
        token_len = len(tokens)
        return tokens, token_len
