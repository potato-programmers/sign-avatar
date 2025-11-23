import json
import torch

class GlossVocab:
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.gloss2id = json.load(f)
        self.id2gloss = {v: k for k, v in self.gloss2id.items()}

        self.pad_id = self.gloss2id["<pad>"]
        self.bos_id = self.gloss2id["<bos>"]
        self.eos_id = self.gloss2id["<eos>"]
        self.unk_id = self.gloss2id["<unk>"]

    def encode(self, gloss_seq, max_len):
        ids = [self.bos_id]
        for g in gloss_seq:
            ids.append(self.gloss2id.get(g, self.unk_id))
        ids.append(self.eos_id)

        length = len(ids)
        if length > max_len:
            ids = ids[:max_len]
            length = max_len

        if len(ids) < max_len:
            ids += [self.pad_id] * (max_len - len(ids))

        return torch.LongTensor(ids), length
