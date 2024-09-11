import argparse
import jsonlines
from tqdm import tqdm
from BCEmbedding import RerankerModel

# your query and corresponding passages
query = 'input_query'
passages = ['passage_0', 'passage_1', ...]

# construct sentence pairs
sentence_pairs = [[query, passage] for passage in passages]


args = argparse.ArgumentParser()
args.add_argument("-model", required=False, default="/home/guodewen/models/bce-reranker-base_v1", type=str)
args.add_argument("-src", required=True, type=str)
args.add_argument("-dst", required=True, type=str)
args.add_argument("-batch", required=False, type=int, default=48)
args = args.parse_args()

# init reranker model
model = RerankerModel(model_name_or_path=args.model, device="cuda:0")

with jsonlines.open(args.src) as fr:
    trainset = list(fr)

# with jsonlines.open(args.dst, "w", flush=True) as fw:
#     for pos in tqdm(range(0, len(trainset), args.batch), total= len(trainset) // args.batch + 1, desc="scoring..."):
#         batch = trainset[pos: pos + args.batch]
#         inputs = [[item["txt1"], item["txt2"]] for item in batch]

#         scores = model.compute_score(inputs)
#         for idx, score in enumerate(scores):
#             if score > 0.35:
#                 fw.write(batch[idx])


with jsonlines.open(args.dst, "w", flush=True) as fw:
    for line in tqdm(trainset):
        inputs = [[line["txt1"], line["txt2"]]] + [[line["txt1"], item] for item in line["hard_negs"]]
        scores = model.compute_score(inputs)
        if scores[0] < 0.35 or any([sco > 0.9 for sco in scores[1:]]): continue
        fw.write(line)
