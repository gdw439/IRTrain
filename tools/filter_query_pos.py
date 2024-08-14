import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('/home/guodewen/research/IRTrain/models/bge-large-reranker')
model = AutoModelForSequenceClassification.from_pretrained('/home/guodewen/research/IRTrain/models/bge-large-reranker')
model = model.to('cuda')
model.eval()

# pairs = [['北戴河有几个', '北戴河有2个'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
# with torch.no_grad():
#     inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
#     print(model(**inputs, return_dict=True).logits)
#     scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
#     print(scores)

# import argparse
import jsonlines

dst_data = []
with jsonlines.open("/home/guodewen/research/IRTrain/dataset/airpline/airplane_train.jsonl") as f:
    for line in f:
        query, document = line['txt1'], line['txt2']
        with torch.no_grad():
            pairs = [[query, document]]
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs = inputs.to('cuda')
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        if scores < 0: continue

        hard_negs = [[query, item] for item in line['hard_negs']]
        with torch.no_grad():
            inputs = tokenizer(hard_negs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs = inputs.to('cuda')
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        
        new_hard = [docs for score, docs in zip(scores, line['hard_negs']) if score < 0]
        line['hard_negs'] = new_hard

        dst_data.append(line)


with jsonlines.open("/home/guodewen/research/IRTrain/dataset/airpline/airplane_train-filter-2.jsonl", 'w') as f:
    f.write_all(dst_data)