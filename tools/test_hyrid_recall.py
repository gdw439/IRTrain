import random
import string
import numpy as np

from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from pymilvus import (
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection, AnnSearchRequest, RRFRanker, connections,
)

import logging

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 DEBUG（包括 DEBUG 以上的所有日志）
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日志格式
    datefmt='%Y-%m-%d %H:%M:%S',  # 日期时间格式
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('app.log')  # 输出到文件
    ]
)



class HybridIndex(object):
    def __init__(self, model_name='/Users/dwguo/Downloads/bge-m3', dense_dim=1024, col_name = 'hybrid3') -> None:
        # 2. setup Milvus collection and index
        connections.connect("default", host="localhost", port="19530")
        self.func = BGEM3EmbeddingFunction(model_name=model_name, use_fp16=False, device="mps")

        # Specify the data schema for the new Collection.
        fields = [
            # Use auto generated id as primary key
            FieldSchema(name="pk", dtype=DataType.VARCHAR,
                        is_primary=True, auto_id=True, max_length=100),
            # Store the original text to retrieve based on semantically distance
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65530),
            # Milvus now supports both sparse and dense vectors, we can store each in
            # a separate field to conduct hybrid search on both vectors.
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR,
                        dim=dense_dim),
        ]
        schema = CollectionSchema(fields, "")
        # Now we can create the new collection with above name and schema.
        self.col = Collection(col_name, schema, consistency_level="Strong")

        # We need to create indices for the vector fields. The indices will be loaded
        # into memory for efficient search.
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        self.col.create_index("sparse_vector", sparse_index)
        dense_index = {"index_type": "FLAT", "metric_type": "IP"}
        self.col.create_index("dense_vector", dense_index)
        self.col.load()
    

    def insert(self, contents, batch = 64):
        from tqdm import tqdm
        for pos in tqdm(range(0, len(contents), batch)):
            docs_embeddings = self.func.encode_documents(contents[pos:pos + batch])
            entities = [contents[pos:pos + batch], docs_embeddings["sparse"], docs_embeddings["dense"]]
            self.col.insert(entities)
        self.col.flush()


    def search(self, querys, topn = 5):
        query_embeddings = self.func.encode_queries(querys)
        # Prepare the search requests for both vector fields
        sparse_search_params = {"metric_type": "IP"}
        sparse_req = AnnSearchRequest(query_embeddings["sparse"],
                                    "sparse_vector", sparse_search_params, limit=topn)
        dense_search_params = {"metric_type": "IP"}
        dense_req = AnnSearchRequest(query_embeddings["dense"],
                                    "dense_vector", dense_search_params, limit=topn)

        # Search topK docs based on dense and sparse vectors and rerank with RRF.
        res = self.col.hybrid_search([sparse_req, dense_req], rerank=RRFRanker(),
                                limit=topn, output_fields=['text'])
        # print("res: ", res)
        result_texts = [hit.fields["text"] for hit in res[0]]
        return result_texts

# hybrid = HybridIndex()
# hybrid.insert(['合肥天气怎么样', '合肥天气', '北京市政府在哪', '合肥', 'dad'])
# print(hybrid.search(['合肥'], topn=3))

if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    # args.add_argument('-m', type=str, required=True, help='model path')
    args.add_argument('-q', type=str, required=True, help='query file with answer phase')
    args.add_argument('-c', type=str, required=True, help='corpus file')
    args.add_argument('-t', type=int, required=True, help='topn')
    args = args.parse_args()

    index = HybridIndex()

    qd_pair = {}
    import jsonlines
    with jsonlines.open(args.q, 'r') as f:
        for i in f:
            qd_pair[i['query']] = qd_pair.get(i['query'], set())
            qd_pair[i['query']].add(i['content'])

    # with jsonlines.open(args.q, 'r') as f:
    #     for i in f:
    #         # if i['question_type'] != '事实性问题': continue
    #         qd_pair[i['question']] = qd_pair.get(i['question'], set())
    #         qd_pair[i['question']].add(i['article'].replace('\n', ''))

    with jsonlines.open(args.c, 'r') as f:
        corpus = [i['content'].replace("\n", "").strip() for i in f]
        from collections import OrderedDict
        corpus = list(OrderedDict.fromkeys(corpus))

    # corpus = list(set(corpus) | set([cor for corpus in qd_pair.values() for cor in corpus]))
    
    corpus = [cor for cor in corpus if cor != '']

    index.insert(corpus)


    # import numpy as np
    # lens = []
    # cnt = 0
    # for cor in corpus:
    #     if len(cor) > 512:
    #         cnt += 1
    #     lens.append(len(cor))
    # print(f"len>512 % {cnt / len(lens)}")
    # print(f"average len: {np.mean(lens)}")


    qs = [q for q, _ in qd_pair.items()]
    cnt, batch = 0, 256

    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['query', 'content', 'label'] + [f'top{i}' for i in range(5)])
    for q in qs:
        ans = index.search([q], args.t)
        # ca = [a['content'] for a in ans]
        ca = ans

        qr = qd_pair[q] 
        cnt = cnt + 1 if qr & set(ca) else cnt
        ws.append( [q, '\n\n'.join(list(qr)), True if qr & set(ca) else False] + ca )
    wb.save("topn.xlsx")
    print(f'recall of top {args.t} is {cnt / len(qs) :.2%}')