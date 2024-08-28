import os
import torch
import numpy as np
from typing import List
from transformers import AutoTokenizer, BertModel
import json
import codecs
import openpyxl
from functools import reduce

class ModelEmbed(object):
    ''' 使用模型将文本表征为向量
    '''
    def __init__(self, path, device='cuda', batch_size = 256) -> None:
        self.token = AutoTokenizer.from_pretrained(path)
        self.model = BertModel.from_pretrained(path, device_map=device)
        self.model = self.model.half()
        self.device = self.model.device
        self.batch_size = batch_size

    def emdbed(self, text: List[str], max_length: int = 1024) -> torch.Tensor :
        if isinstance(text, str): text = [text]
        
        vector_buff = []
        # 当batch size 过大的时候分批计算
        for pos in range(0, len(text), self.batch_size):
            prein = self.token.batch_encode_plus(
                text[pos: pos + self.batch_size], 
                padding="longest", 
                truncation=True, 
                max_length=max_length, 
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                model_output = self.model(**prein)
                attention_mask = prein['attention_mask']
                last_hidden = model_output.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
                vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                vectors = torch.nn.functional.normalize(vectors, 2.0, dim=1)
            vector_buff.append(vectors.to('cpu'))
        vectors = torch.vstack(vector_buff)
        return vectors.to('cuda')


class BruteIndex(object):
    def __init__(self, device="cpu") -> None:
        self.device = device
        self.text, self.index = [], []

    def insert(self, text: List[str], embed: torch.Tensor): 
        self.text.extend(text)
        self.index.append(embed)
    
    @torch.no_grad()
    def search(self, embed: torch.Tensor, topn: int = 5, step: int = 256):
        if isinstance(self.index, torch.Tensor) :
            self.index = self.index 
        else:
            self.index = torch.vstack(self.index)

        embed.to(self.device)
        self.index.to(self.device)
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
        if os.path.exists(file_path):
            raise ValueError("file exist and not empty!")
        text_file = os.path.join(file_path, "content.npy")
        index_file = os.path.join(file_path, 'index.bin')
        if os.path.exists(text_file) or os.path.exists(index_file):
            raise ValueError(f'file exist for {text_file} or {index_file}')
        
        np.save(text_file, self.text)
        torch.save(self.index, index_file)


    def dump(self, file_path):
        self.index = self.index if isinstance(self.index, torch.Tensor) else torch.vstack(self.index)
        text_file = os.path.join(file_path, 'content.npy')
        index_file = os.path.join(file_path, "index.bin")
        if not os.path.exists(text_file) or not os.path.exists(index_file):
            raise ValueError(f"please check path {text_file} and {index_file} existable")
        
        self.text = np.load(text_file).tolist()
        self.index = torch.load(index_file)

if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('-m', type=str, required=True, help='model path')
    args.add_argument('-q', type=str, required=True, help='query file with answer phase')
    args.add_argument('-c', type=str, required=True, help='corpus file')
    args.add_argument('-t', type=int, required=True, help='topn recall')
    args.add_argument('-s', type=str, required=True, help='where to save result')
    args = args.parse_args()

    encoder = ModelEmbed(args.m, device='cuda', batch_size=512)
    index = BruteIndex(device='cuda')

    qd_pair = {}
    qs = []
    import jsonlines
    with jsonlines.open(args.q, 'r') as f:
        for i in f:
            qs.append(i['query'])
            qd_pair[i['query']] = qd_pair.get(i['query'], [])
            qd_pair[i['query']].append(i['content'])

    contents = []
    slice2content = {}
    with jsonlines.open(args.c, 'r') as f:
        slice_list = list(f)
        for idx, item in enumerate(slice_list):
            content = item['content']
            slices = item['slices']
            
            # 切片太短说明有问题，解析出来非json也有问题，这种情况下就不切分
            if not isinstance(slices, list) or min([len(item) for item in slices]) < 32:
                if content in slice2content:
                    slice2content[content].append(content)
                else :
                    slice2content[content] = [content]
                contents.append(content)
                continue

            contents.extend(slices)
            for sli in slices:
                if sli in slice2content:
                    slice2content[sli].append(content)
                else:
                    slice2content[sli] = [content]

    from collections import OrderedDict
    corpus = list(OrderedDict.fromkeys(contents))
    corpus_emb = encoder.emdbed(corpus)
    print("corpus_emb.shape: ", corpus_emb.shape)
    index.insert(corpus, corpus_emb)

    cnt, batch = 0, 512
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['query', 'content', 'label'] + [f'top{i}' for i in range(args.t)])
    for p in range(0, len(qs), batch):
        querys = qs[p: p+batch]
        if len(querys) == 0: continue
        qb_emb = encoder.emdbed(querys)
        scores, values = index.search(qb_emb, topn=6477)
        
        for query, score_t, recall_t in zip(querys, scores, values):
            ground_truth_contents = qd_pair[query]
            contents = {}
            for idx, (score, item) in enumerate(zip(score_t, recall_t)):
                content_set = slice2content[item]
                for content in content_set:
                    if content in contents:
                        contents[content] = max(score, contents[content])
                        # contents[content] += 1 / (61 + idx)
                    else:
                        contents[content] = score
                        # contents[content] = 1 / (61 + idx)
            contents = [(key, val) for key, val in contents.items()]
            contents = sorted(contents, key= lambda x: x[1])
            recall_map2_contents = [item[0] for item in contents[-args.t:]]
            # recall_map2_contents = set(recall_map2_contents)
            # recall_map2_contents = reduce(set.union, [slice2content[item] for item in recall_t])

            label = False
            for a in ground_truth_contents:
                for b in recall_map2_contents:
                    if a == b:
                        label = True
            # label = len(ground_truth_contents & recall_map2_contents) > 0
            cnt = cnt + 1 if label else cnt
            ws.append( [query, '\n\n'.join(list(ground_truth_contents)), label] + list(recall_map2_contents))
    wb.save(f"{args.s}/top-chunk{args.t}.xlsx")

    print(f'recall {cnt} / {len(qs)} is {cnt / len(qs) :.2%}')