from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus.model.sparse import BM25EmbeddingFunction


collection_name = "hello_sparse"
milvus_client = MilvusClient("http://localhost:19530")

has_collection = milvus_client.has_collection(collection_name, timeout=5)
if has_collection:
    milvus_client.drop_collection(collection_name)

fields = [
    FieldSchema(name="idx", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65530),
    FieldSchema(name="sparse_embed", dtype=DataType.SPARSE_FLOAT_VECTOR),
]
schema = CollectionSchema(fields, "demo for using sparse float vector with milvus client")
index_params = milvus_client.prepare_index_params()

index_params.add_index(field_name="sparse_embed", index_name="sparse_inverted_index", index_type="SPARSE_INVERTED_INDEX", metric_type="IP", params={"drop_ratio_build": 0.2})
milvus_client.create_collection(collection_name, schema=schema, index_params=index_params, timeout=5)

import jsonlines

document = []
with jsonlines.open('/Users/dwguo/Desktop/project/4.download_milvus_repo/bge_large_0_0.jsonl') as f:
    for item in f:
        item = item['content'].replace("\n", "").strip()
        document.append(item)

analyzer = build_default_analyzer(language="zh")
bm25_ef = BM25EmbeddingFunction(analyzer, corpus=document, num_workers=1)

rows = []
sparse = bm25_ef.encode_documents(document)
for doc, spa in zip(document, sparse):
    spa = spa.tocoo()
    spa = {int(k): float(v) for k, v in zip(spa.col, spa.data)}
    if len(spa) == 0:
        print(doc)
        continue
    rows.append({"content": doc, "sparse_embed": spa})
insert_result = milvus_client.insert(collection_name, rows, progress_bar=True)
# print(insert_result)


vectors_to_search = bm25_ef.encode_queries(["清华大学"])
search_params = {
    "metric_type": "IP",
    "params": {
        "drop_ratio_search": 0.2,
    }
}
# print(vectors_to_search)
# import time
# for _ in range(16):
#     start = time.perf_counter()
#     # no need to specify anns_field for collections with only 1 vector field
#     result = milvus_client.search(collection_name, [vectors_to_search], limit=3, output_fields=["content"], search_params=search_params)
#     ends = time.perf_counter()
#     print(ends - start)
from tqdm import tqdm
cnt, total = 0, 0
import time
time.sleep(3)
time_fly = []
with jsonlines.open("test.jsonl") as f:
    for q in tqdm(f):
        start = time.perf_counter()
        vectors_to_search = bm25_ef.encode_queries([q['query']])
        print(vectors_to_search)
        result = milvus_client.search(collection_name, [vectors_to_search], limit=5, output_fields=["content"], search_params=search_params)
        ends = time.perf_counter()
        time_fly.append(ends - start)
        docs = [r['entity']['content'] for res in result for r in res]
        if q['content'] in set(docs):
            cnt += 1
        total += 1
print(cnt, total, cnt / total)
import numpy as np
print(np.mean(time_fly))
# vectors_to_search = bm25_ef.encode_queries(["清华大学 校医院"])
# result = milvus_client.search(collection_name, [vectors_to_search], limit=5, output_fields=["content"], search_params=search_params)
# print([r['entity']['content'] for res in result for r in res])
        
        



# bm25_ef.fit(["你好"])
# bm25_ef.save("token.json")
milvus_client.drop_collection(collection_name)