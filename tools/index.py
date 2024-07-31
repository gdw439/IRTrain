import os
import torch
import numpy as np
from typing import List
from transformers import AutoTokenizer, BertModel


class ModelEmbed(object):
    ''' 使用模型将文本表征为向量
    '''
    def __init__(self, path, device='cuda', batch_size = 256) -> None:
        self.token = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = BertModel.from_pretrained(path, trust_remote_code=True, device_map=device)
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


from FlagEmbedding import BGEM3FlagModel

class BGEM3(object):
    ''' 使用模型将文本表征为向量
    '''
    def __init__(self, path, device='cuda', batch_size = 256) -> None:
        self.token = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = BGEM3FlagModel(path, use_fp16=True, device=device)
        self.device = self.model.device
        self.batch_size = batch_size

    def emdbed(self, text: List[str], max_length: int = 1024) -> torch.Tensor :
        print(text[:5])
        text = ['..' if item == '' else item for item in text]
        print(len(text))
        vectors = self.model.encode(text, batch_size=self.batch_size, max_length=8192)['dense_vecs']
        return torch.tensor(vectors)


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
            # for item in self.index:
            #     item = item.to('cpu')
            self.index = self.index[0]
            print(self.index.shape)
            
            # torch.vstack(self.index)

        embed.to(self.device)
        self.index.to(self.device)

        # print(self.index.shape)
        # print(len(self.text))
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


def load_tsv(file_path):
    import csv, sys
    csv.field_size_limit(sys.maxsize)
    with open(file_path, newline='', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader)
        for [idx, text] in reader:
            yield text


# if __name__ == '__main__':
    # cache = []
    # embed = ModelEmbed('/home/guodewen/research/stella/model')
    # index = BruteIndex()
    # for idx, text in enumerate(load_tsv("/home/guodewen/research/stella/dataset/collection.tsv")):
    #     cache.append(text)
    #     if idx % 256 == 0:
    #         vecs = embed.emdbed(cache)
    #         index.insert(cache, vecs)
    #         cache = []

    #     if idx == 256:
    #         break

    # ans = index.search(embed.emdbed(["本人大一新生"]))
    # print(ans)


def chunks(datain):
    import re
    pattern = re.compile(r'([^。！？]*[。！？]|[^。！？]+$)')

    spans = []
    spans2idx = {}
    # 使用findall方法找到所有匹配的文本段
    for idx, text in enumerate(datain):
        spans2idx[text] = spans2idx.get(text, set())
        spans2idx[text].add(idx)
        spans.append(text)
        # ans = list(pattern.findall(text))

        # tmp = ''
        # chk = []
        # for an in ans:
        #     if len(tmp + an) >= 512:
        #         chk.append(tmp)
        #         tmp = ''
        #     tmp += an
        # if tmp != "":
        #     chk.append(tmp)

        # ans = chk
        # spans.extend(ans)
        # for an in ans:
        #     spans2idx[an] = spans2idx.get(an, set())
        #     spans2idx[an].add(idx)
    return spans, spans2idx


if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('-m', type=str, required=True, help='model path')
    args.add_argument('-q', type=str, required=True, help='query file with answer phase')
    args.add_argument('-c', type=str, required=True, help='corpus file')
    args.add_argument('-t', type=int, required=True, help='topn recall')
    args = args.parse_args()

    # encoder = ModelEmbed(args.m, device='cuda', batch_size=512)
    encoder = BGEM3(args.m, device='cuda', batch_size=16)
    index = BruteIndex(device='cuda')

    qd_pair = {}
    import jsonlines
    # with jsonlines.open(args.q, 'r') as f:
    #     for i in f:
    #         qd_pair[i['query']] = qd_pair.get(i['query'], set())
    #         qd_pair[i['query']].add(i['content'])
    with jsonlines.open(args.q, 'r') as f:
        for i in f:
            if i['question_type'] != '事实性问题': continue
            qd_pair[i['question']] = qd_pair.get(i['question'], set())
            qd_pair[i['question']].add(i['article'].replace('\n', ''))

    with jsonlines.open(args.c, 'r') as f:
        corpus = [i['content'].replace("\n", "").strip() for i in f]
        from collections import OrderedDict
        corpus = list(OrderedDict.fromkeys(corpus))

    # 确保召回的文档确实在index中
    corpus = list(set(corpus) | set([cor for corpus in qd_pair.values() for cor in corpus]))

    span, tabs = chunks(corpus)
    print(span[:5])
    corpus_emb = encoder.emdbed(span)
    print("corpus_emb.shape: ", corpus_emb.shape)
    index.insert(span, corpus_emb)

    qs = [q for q, _ in qd_pair.items()]
    cnt, batch = 0, 512

    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['query', 'content', 'label'] + [f'top{i}' for i in range(args.t)])
    for p in range(0, len(qs), batch):
        qb = qs[p: p+batch]
        if len(qb) == 0: continue
        qb_emb = encoder.emdbed(qb)
        score, value = index.search(qb_emb, topn=args.t)
        
        for q, sc, ca in zip(qb, score, value):
            qr = qd_pair[q]

            corpus_2_score = {}
            for s, c in zip(sc, ca):
                for cid in tabs[c]:
                    corpus_2_score[corpus[cid]] = max(corpus_2_score.get(corpus[cid], 0), s)
            topn = sorted(list(corpus_2_score.items()), key=lambda x:x[1])
            topn = [i[0] for i in topn[-args.t:]] 

            cnt = cnt + 1 if len(qr & set(topn)) > 0 else cnt
            ws.append( [q, '\n\n'.join(list(qr)), len(qr & set(topn))] + topn)
    wb.save("topn.xlsx")

    print(f'recall {cnt} / {len(qs)} is {cnt / len(qs) :.2%}')
