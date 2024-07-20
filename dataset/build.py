import csv
import sys
import json
import codecs
csv.field_size_limit(sys.maxsize)

handle = codecs.open("t2ranking_dev.jsonl", "w", encoding="utf8")

paras_tab = {}
query_tab = {}
print("start deal with train data...")

file_path = 'collection.tsv'
with open(file_path, newline='', encoding='utf-8') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    next(reader)
    for [idx, paras] in reader:
        paras_tab[idx] = paras


file_path = 'data_queries.dev.tsv'
with open(file_path, newline='', encoding='utf-8') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    next(reader)
    for [idx, query] in reader:
        query_tab[idx] = query


file_path = 'qrels.retrieval.dev.tsv'
with open(file_path, newline='', encoding='utf-8') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    next(reader)
    for [idq, idp] in reader:
        data = {"txt1": query_tab[idq], "txt2": paras_tab[idp],}
        data = json.dumps(data, ensure_ascii=False)
        handle.write(f"{data}\n")
handle.close()