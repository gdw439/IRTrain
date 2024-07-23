import torch
import numpy as np
from typing import List
from transformers import AutoTokenizer, BertModel


class ModelEmbed(object):
    def __init__(self, path, device='auto', batch_size = 256) -> None:
        self.token = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = BertModel.from_pretrained(path, trust_remote_code=True, device_map=device)
        self.device = self.model.device
        self.batch_size = batch_size

    def emdbed(self, text):
        if isinstance(text, str):
            text = [text]
        
        vector_buff = []
        for pos in range(0, len(text), self.batch_size):
            prein = self.token.batch_encode_plus(
                text[pos: pos + self.batch_size], 
                padding="longest", 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(self.device)

            # print(prein)
            with torch.no_grad():
                model_output = self.model(**prein)
                attention_mask = prein['attention_mask']
                last_hidden = model_output.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
                vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                vectors = torch.nn.functional.normalize(vectors, 2.0, dim=1)
            vector_buff.append(vectors)
        vectors = torch.vstack(vector_buff)
        return vectors


class BruteIndex(object):
    def __init__(self, device="cpu") -> None:
        self.device = device
        self.text, self.index = [], []

    def insert(self, text: List[str], embed: torch.Tensor): 
        self.text.extend(text)
        self.index.append(embed)
    

    def search(self, embed: torch.Tensor, topn: int = 5, step: int = 2560):
        embed.to(self.device)
        self.index.to(self.device)

        self.index = torch.vstack(self.index)
        print(self.index.shape)
        print(len(self.text))
        assert self.index.shape[0] == len(self.text), "length not same"

        score_buff = []
        for idb in range(0, len(self.text), step):
            # cache = [step, dims]
            cache = self.index[idb: idb + step, :] 

            # step_score = [batch, step]
            step_score = embed @ cache.T
            score_buff.append(step_score)

        # batch_score = [batch, len(self.text)]
        batch_score = torch.hstack(score_buff)

        # batch_score = [batch, topn]
        values, indices = torch.topk(batch_score, topn, dim=-1)

        content = [[self.text[row] for row in col] for col in indices.tolist()]
        return values.tolist(), content


    def load(self, file_path):
        pass


    def dump(self, file_path):
        pass



def load_tsv(file_path):
    import csv, sys
    csv.field_size_limit(sys.maxsize)
    with open(file_path, newline='', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader)
        for [idx, text] in reader:
            yield text


if __name__ == '__main__':
    cache = []
    embed = ModelEmbed('/home/guodewen/research/stella/model')
    index = BruteIndex()
    for idx, text in enumerate(tsv_load("/home/guodewen/research/stella/dataset/collection.tsv")):
        cache.append(text)
        if idx % 256 == 0:
            vecs = embed.emdbed(cache)
            index.insert(cache, vecs)
            cache = []

        if idx == 256:
            break

    ans = index.search(embed.emdbed(["本人大一新生"]))
    print(ans)
        
            