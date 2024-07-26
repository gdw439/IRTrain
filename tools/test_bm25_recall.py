import argparse
args = argparse.ArgumentParser()
# args.add_argument('-m', type=str, required=True, help='model path')
args.add_argument('-q', type=str, required=True, help='query file with answer phase')
args.add_argument('-c', type=str, required=True, help='corpus file')
args = args.parse_args()

from bm25_retriever.bm25_retriever import BM25Index
index = BM25Index()

qd_pair = {}
import jsonlines
with jsonlines.open(args.q, 'r') as f:
    for i in f:
        qd_pair[i['query']] = qd_pair.get(i['query'], set())
        qd_pair[i['query']].add(i['content'])

with jsonlines.open(args.c, 'r') as f:
    corpus = [i['content'].replace("\n", "").strip() for i in f]
    from collections import OrderedDict
    corpus = list(OrderedDict.fromkeys(corpus))

index.build(corpus)
index.load()

import numpy as np
lens = []
cnt = 0
for cor in corpus:
    if len(cor) > 512:
        cnt += 1
    lens.append(len(cor))
print(f"len>512 % {cnt / len(lens)}")
print(f"average len: {np.mean(lens)}")


qs = [q for q, _ in qd_pair.items()]
cnt, batch = 0, 256

import openpyxl
wb = openpyxl.Workbook()
ws = wb.active
ws.append(['query', 'content', 'label'] + [f'top{i}' for i in range(5)])
for q in qs:
    ans = index.search(q, 5)
    ca = [a['content'] for a in ans]

    qr = qd_pair[q] 
    cnt = cnt + 1 if qr & set(ca) else cnt
    ws.append( [q, '\n\n'.join(list(qr)), True if qr & set(ca) else False] + ca )
wb.save("topn.xlsx")
print(f'recall is {cnt / len(qs) :.2%}')