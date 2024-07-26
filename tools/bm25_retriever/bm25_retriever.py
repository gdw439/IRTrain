from whoosh.fields import *
from whoosh.index import create_in, open_dir
from whoosh.query import compound, Term
from whoosh.qparser import QueryParser, SimpleParser

import os, json, codecs
from pathlib import Path
from bm25_retriever.analyzer import ChineseAnalyzer

index_path = (Path(__file__).resolve().parent / 'index').as_posix()
# cite https://github.com/goto456/stopwords.git
stopword_path = (Path(__file__).resolve().parent / 'baidu_stopwords.txt').as_posix()

if not os.path.exists(index_path):
    print(f"no found path {index_path}, mkdir one")
    os.mkdir(index_path)


def get_analyzer():
    if os.path.exists(stopword_path):
        stopword = codecs.open(stopword_path, "r", encoding="utf8").readlines()
        stopword = [item.rstrip('\n') for item in stopword]
        stopword = frozenset(stopword)
        return ChineseAnalyzer(stoplist=stopword)
    else:
        return ChineseAnalyzer()

# class ArticleSchema(SchemaClass):
#     start_index = NUMERIC(stored=True, numtype=int)
#     knowledge_name = TEXT(stored=True)
#     file_id = NUMERIC(stored=True, numtype=int)
#     content = TEXT(stored=True)
#     segment_id = NUMERIC(stored=True, numtype=int)
#     knowledge_id = NUMERIC(stored=True, numtype=int)
#     span = TEXT(stored=True, analyzer=get_analyzer())
#     id = TEXT(stored=True)
#     end_index = NUMERIC(stored=True, numtype=int)
#     full_title = TEXT(stored=True)


class ArticleSchema(SchemaClass):
    content = TEXT(stored=True, analyzer=get_analyzer())


class BM25Index(object):
    def __init__(self) -> None:
        schema = ArticleSchema()
        self.ix = create_in(index_path, schema, indexname='demo')

    def build(self, data):
        writer = self.ix.writer()

        for d in data:
            writer.add_document(content= d)
        writer.commit()


    def load(self):
        self.whoosh = open_dir(index_path, indexname='demo')


    def search(self, query: str, topn: int) :
        with self.whoosh.searcher() as searcher:
            query = SimpleParser("content", self.whoosh.schema).parse(query)
            results = searcher.search(query, limit=topn)
            return [{"score": res.score, **res.fields()} for res in results]
