import argparse
args = argparse.ArgumentParser()
# args.add_argument('-m', type=str, required=True, help='model path')
args.add_argument('-q', type=str, required=True, help='query file with answer phase')
args.add_argument('-c', type=str, required=True, help='corpus file')
args.add_argument('-t', type=int, required=True, help='topn')
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

corpus = list(set(corpus) | set([cor for corpus in qd_pair.values() for cor in corpus]))

index.build(corpus)
index.load()

qs = [q for q, _ in qd_pair.items()]
cnt, batch = 0, 256

import time
import openpyxl
time_fly = []
wb = openpyxl.Workbook()
ws = wb.active
ws.append(['query', 'content', 'label'] + [f'top{i}' for i in range(5)])
for q in qs:
    a = time.perf_counter()
    ans = index.search(q, args.t)
    ca = [a['content'] for a in ans]
    b = time.perf_counter()

    time_fly.append(b - a)

    qr = qd_pair[q] 
    cnt = cnt + 1 if qr & set(ca) else cnt
    ws.append( [q, '\n\n'.join(list(qr)), True if qr & set(ca) else False] + ca )
wb.save(f"top{args.t}.xlsx")
print(f'recall of top {args.t} is {cnt / len(qs) :.2%}')

import numpy as np
print(f"time spend: {np.mean(time_fly)} s")