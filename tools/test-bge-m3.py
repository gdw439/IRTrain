import logging

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 DEBUG（包括 DEBUG 以上的所有日志）
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日志格式
    datefmt='%Y-%m-%d %H:%M:%S',  # 日期时间格式
    handlers=[logging.StreamHandler()]
)

model_name = '/home/guodewen/research/IRTrain/models/bge-m3'
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
func = BGEM3EmbeddingFunction(model_name=model_name, use_fp16=True, device="cuda")
from scipy.sparse import csr_matrix, vstack, save_npz, load_npz
import jsonlines
import numpy as np

with jsonlines.open('/home/guodewen/research/IRTrain/dataset/soda_stella/bge_large_0_0.jsonl', 'r') as f:
    corpus = [i['content'].replace("\n", "").strip() for i in f]
    from collections import OrderedDict
    corpus = list(OrderedDict.fromkeys(corpus))

corpus = [cor for cor in corpus if cor != '']

# dense_data, sparse_data = [], []
# batch = 102
# for p in range(0, len(corpus), batch):
#     indata = corpus[p: p + batch]
#     ans = func(indata)
#     dense_data.append(ans['dense'])
#     sparse_data.append(ans['sparse'])

# dense_data = np.vstack(dense_data)
# sparse_data = vstack(sparse_data)
ans = func(corpus)
dense_data = ans['dense']
sparse_data = ans['sparse']
np.save('corpus.dense', dense_data)
save_npz('corpus.sparse', sparse_data)


qd_pair = {}
import jsonlines
with jsonlines.open('/home/guodewen/research/IRTrain/dataset/soda_stella/test.jsonl', 'r') as f:
    for i in f:
        qd_pair[i['query']] = qd_pair.get(i['query'], set())
        qd_pair[i['query']].add(i['content'])
qs = [q for q, _ in qd_pair.items()]
ans = func(qs)
dense_data = ans['dense']
sparse_data = ans['sparse']

np.save('query.dense', dense_data)
save_npz('query.sparse', sparse_data)
