import jieba
from pyserini.index import IndexWriter
from pyserini.search import SimpleSearcher

# 准备没有ID的中文文档
docs = [
    '这是第一篇文档的内容。',
    '这是第二篇文档的内容。',
    '这是第三篇文档的内容。'
]

# 创建索引
index_writer = IndexWriter('index_path')

# 为每个文档生成唯一的ID并进行分词
for idx, content in enumerate(docs):
    doc_id = f'doc{idx+1}'  # 生成唯一ID
    tokenized_content = ' '.join(jieba.cut(content))  # 分词
    index_writer.add_document(doc_id, tokenized_content)

index_writer.close()

# 初始化搜索器
searcher = SimpleSearcher('index_path')

# 执行查询
query = '文档内容'
hits = searcher.search(query)

# 输出结果
for i in range(len(hits)):
    print(f'{i+1:2} {hits[i].docid:15} {hits[i].score:.5f}')